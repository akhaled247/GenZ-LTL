from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch

from torch_ac.utils import DictList
from torch_ac.utils.action_bridge import make_action_bridge
from preprocessing.ma_layout import flatten_ma_obs, unflatten_ma_actions
from preprocessing.preprocessing_ma import preprocess_obss_ma


def _agent_keys(env) -> list[str]:
    if hasattr(env, "agent_ids"):
        return list(env.agent_ids)
    return list(env.unwrapped.possible_agents)


def _agent_action_space(env, agent_key: str):
    space = env.action_space
    if callable(space):
        return space(agent_key)
    return space[agent_key]


class BaseAlgoMA(ABC):
    """On-policy MA base: merged buffer T * n_envs * n_agents with per-agent critics."""

    def __init__(
            self, envs, model, device, num_steps_per_proc, discount, lr, gae_lambda, entropy_coef,
            value_loss_coef, max_grad_norm, recurrence=1, fast_action_bridge=False,
    ):
        from envs.vec.ma_sync_env import MASyncVecEnv

        self.agent_keys = _agent_keys(envs[0])
        self.num_agents = len(self.agent_keys)
        self.num_envs = len(envs)
        self.env = MASyncVecEnv(envs)
        self.propositions = self.env.get_propositions()
        self.model = model
        self.device = device
        self.num_steps_per_proc = num_steps_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence

        assert self.model.recurrent or self.recurrence == 1
        assert self.num_steps_per_proc % self.recurrence == 0

        self.model.to(self.device)
        self.model.train()

        self.num_procs = self.num_envs * self.num_agents
        self.num_steps = self.num_steps_per_proc * self.num_procs
        probe_space = _agent_action_space(envs[0], self.agent_keys[0])
        self.action_space_shape = probe_space.shape
        bridge_shape = (self.num_procs,) + tuple(self.action_space_shape)
        self.action_bridge = make_action_bridge(device, bridge_shape, fast_action_bridge)

        shape = (self.num_steps_per_proc, self.num_procs)
        act_shape = shape + self.action_space_shape

        self.obs = self.env.reset()
        self.obss = [None] * shape[0]
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*act_shape, device=self.device)
        self.values = torch.zeros(*shape, device=self.device)
        self.qs = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        self.agent_ids_buf = torch.zeros(*shape, dtype=torch.long, device=self.device)

        self.log_episode_return = torch.zeros(self.num_envs, device=self.device)
        self.log_episode_num_steps = torch.zeros(self.num_envs, device=self.device)
        self.log_done_counter = 0
        self.log_return = [0] * self.num_envs
        self.log_num_steps = [0] * self.num_envs
        self.log_success = [0] * self.num_envs
        self.log_violation = [0] * self.num_envs
        self.goal_success = defaultdict(int)
        self.goal_counts = defaultdict(int)

    def _flat_obs_and_ids(self):
        flat_obs, agent_ids = flatten_ma_obs(self.obs, self.agent_keys)
        return flat_obs, agent_ids

    def collect_experiences(self):
        for i in range(self.num_steps_per_proc):
            flat_obs, agent_ids = self._flat_obs_and_ids()
            preprocessed = preprocess_obss_ma(
                flat_obs, self.propositions, agent_ids, device=self.device,
            )
            with torch.no_grad():
                dist, value = self.model(preprocessed)
            action = dist.sample()

            env_actions = unflatten_ma_actions(
                action, self.num_envs, self.num_agents, self.agent_keys,
            )
            obs, reward, done, info = self.env.step(env_actions)

            self.obss[i] = self.obs
            self.obs = obs

            reward_arr = np.asarray(reward, dtype=np.float32)
            done_arr = np.asarray(done, dtype=bool)
            team_reward = np.repeat(reward_arr, self.num_agents)
            team_done = np.repeat(done_arr, self.num_agents)

            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(team_done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            self.rewards[i] = torch.tensor(team_reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action).detach()
            self.agent_ids_buf[i] = torch.tensor(agent_ids, device=self.device, dtype=torch.long)

            self.log_episode_return += torch.tensor(reward_arr, device=self.device, dtype=torch.float)
            self.log_episode_num_steps += torch.ones(self.num_envs, device=self.device)

            for j, done_ in enumerate(done_arr):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[j].item())
                    self.log_num_steps.append(self.log_episode_num_steps[j].item())
                    self.log_success.append(int("success" in info[j]))
                    self.log_violation.append(int("violation" in info[j]))
                    goal = self.obss[i][j][self.agent_keys[0]]["initial_goal"]
                    self.goal_success[goal] += int("success" in info[j])
                    self.goal_counts[goal] += 1

            self.log_episode_return *= (1 - torch.tensor(done_arr, device=self.device, dtype=torch.float))
            self.log_episode_num_steps *= (1 - torch.tensor(done_arr, device=self.device, dtype=torch.float))

        flat_obs, agent_ids = self._flat_obs_and_ids()
        preprocessed = preprocess_obss_ma(
            flat_obs, self.propositions, agent_ids, device=self.device,
        )
        with torch.no_grad():
            _, next_value = self.model(preprocessed)

        for i in reversed(range(self.num_steps_per_proc)):
            next_mask = self.masks[i + 1] if i < self.num_steps_per_proc - 1 else self.mask
            next_value_i = self.values[i + 1] if i < self.num_steps_per_proc - 1 else next_value
            next_advantage = self.advantages[i + 1] if i < self.num_steps_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value_i * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask
            self.qs[i] = self.rewards[i] + self.discount * next_value_i * next_mask

        exps = DictList()
        exps.obs = [
            self.obss[i][j // self.num_agents][self.agent_keys[j % self.num_agents]]
            for j in range(self.num_procs)
            for i in range(self.num_steps_per_proc)
        ]
        exps.agent_id = self.agent_ids_buf.transpose(0, 1).reshape(-1)
        exps.action = self.actions.transpose(0, 1).reshape((-1,) + self.action_space_shape)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.qs = self.qs.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        flat_agent_ids = self.agent_ids_buf.transpose(0, 1).reshape(-1).tolist()
        exps.obs = preprocess_obss_ma(exps.obs, self.propositions, flat_agent_ids, device=self.device)

        keep = max(self.log_done_counter, self.num_envs)
        logs = {
            "return_per_episode": self.log_return[-keep:],
            "num_steps_per_episode": self.log_num_steps[-keep:],
            "success_per_episode": self.log_success[-keep:],
            "violation_per_episode": self.log_violation[-keep:],
            "num_steps": self.num_steps,
            "avg_goal_success": {k: float(v) / self.goal_counts[k] for k, v in self.goal_success.items()},
        }
        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_envs:]
        self.log_num_steps = self.log_num_steps[-self.num_envs:]
        self.goal_success = defaultdict(int)
        self.goal_counts = defaultdict(int)
        return exps, logs

    @abstractmethod
    def update_parameters(self, exps):
        pass


class BaseAlgoLagMA(ABC):
    """RCO MA base with per-agent cost critics."""

    def __init__(
            self, envs, model, device, num_steps_per_proc, discount, lr, gae_lambda, entropy_coef,
            value_loss_coef, max_grad_norm, recurrence=1, fast_action_bridge=False,
    ):
        from envs.vec.ma_sync_env import MASyncVecEnv

        self.agent_keys = _agent_keys(envs[0])
        self.num_agents = len(self.agent_keys)
        self.num_envs = len(envs)
        self.env = MASyncVecEnv(envs)
        self.propositions = self.env.get_propositions()
        self.model = model
        self.device = device
        self.num_steps_per_proc = num_steps_per_proc
        self.discount = discount
        self.cost_discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.cost_gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence

        assert self.model.recurrent or self.recurrence == 1
        assert self.num_steps_per_proc % self.recurrence == 0

        self.model.to(self.device)
        self.model.train()

        self.num_procs = self.num_envs * self.num_agents
        self.num_steps = self.num_steps_per_proc * self.num_procs
        probe_space = _agent_action_space(envs[0], self.agent_keys[0])
        self.action_space_shape = probe_space.shape
        bridge_shape = (self.num_procs,) + tuple(self.action_space_shape)
        self.action_bridge = make_action_bridge(device, bridge_shape, fast_action_bridge)

        shape = (self.num_steps_per_proc, self.num_procs)
        act_shape = shape + self.action_space_shape

        self.obs = self.env.reset()
        self.obss = [None] * shape[0]
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*act_shape, device=self.device)
        self.values = torch.zeros(*shape, device=self.device)
        self.cost_values = torch.zeros(*shape, device=self.device)
        self.qs = torch.zeros(*shape, device=self.device)
        self.cost_qs = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.returnn = torch.zeros(*shape, device=self.device)
        self.costs = torch.zeros(*shape, device=self.device)
        self.cost_returnn = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.cost_advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        self.agent_ids_buf = torch.zeros(*shape, dtype=torch.long, device=self.device)

        self.log_episode_return = torch.zeros(self.num_envs, device=self.device)
        self.log_episode_cost_return = torch.zeros(self.num_envs, device=self.device)
        self.log_episode_num_steps = torch.zeros(self.num_envs, device=self.device)
        self.log_done_counter = 0
        self.log_return = [0] * self.num_envs
        self.log_cost_return = [0] * self.num_envs
        self.log_num_steps = [0] * self.num_envs
        self.log_success = [0] * self.num_envs
        self.log_violation = [0] * self.num_envs
        self.goal_success = defaultdict(int)
        self.goal_counts = defaultdict(int)

    def _flat_obs_and_ids(self):
        return flatten_ma_obs(self.obs, self.agent_keys)

    def collect_experiences(self):
        for i in range(self.num_steps_per_proc):
            flat_obs, agent_ids = self._flat_obs_and_ids()
            preprocessed = preprocess_obss_ma(
                flat_obs, self.propositions, agent_ids, device=self.device,
            )
            with torch.no_grad():
                dist, value, cost_value = self.model(preprocessed)
            action = dist.sample()

            env_actions = unflatten_ma_actions(
                action, self.num_envs, self.num_agents, self.agent_keys,
            )
            obs, reward, done, info = self.env.step(env_actions)

            reward_cost = np.asarray(reward, dtype=np.float32)
            reward_arr = reward_cost[:, 0]
            cost_arr = reward_cost[:, 1]
            done_arr = np.asarray(done, dtype=bool)
            new_done = np.logical_or(reward_arr.astype(bool), done_arr)
            team_done = np.repeat(new_done, self.num_agents)

            self.obss[i] = self.obs
            self.obs = obs
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(team_done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            self.cost_values[i] = cost_value
            self.rewards[i] = torch.tensor(np.repeat(reward_arr, self.num_agents), device=self.device)
            self.costs[i] = torch.tensor(np.repeat(cost_arr, self.num_agents), device=self.device)
            self.log_probs[i] = dist.log_prob(action).detach()
            self.agent_ids_buf[i] = torch.tensor(agent_ids, device=self.device, dtype=torch.long)

            self.log_episode_return += torch.tensor(reward_arr, device=self.device, dtype=torch.float)
            log_cost = np.where(cost_arr > 0, 1.0, 0.0)
            self.log_episode_cost_return += torch.tensor(log_cost, device=self.device, dtype=torch.float)
            self.log_episode_num_steps += torch.ones(self.num_envs, device=self.device)

            for j, done_ in enumerate(new_done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[j].item())
                    self.log_cost_return.append(self.log_episode_cost_return[j].item())
                    self.log_num_steps.append(self.log_episode_num_steps[j].item())
                    self.log_success.append(int("success" in info[j]))
                    self.log_violation.append(int("violation" in info[j]))
                    goal = self.obss[i][j][self.agent_keys[0]]["initial_goal"]
                    self.goal_success[goal] += int("success" in info[j])
                    self.goal_counts[goal] += 1

            mask_env = 1 - torch.tensor(new_done, device=self.device, dtype=torch.float)
            self.log_episode_return *= mask_env
            self.log_episode_cost_return *= mask_env
            self.log_episode_num_steps *= mask_env

        flat_obs, agent_ids = self._flat_obs_and_ids()
        preprocessed = preprocess_obss_ma(
            flat_obs, self.propositions, agent_ids, device=self.device,
        )
        with torch.no_grad():
            _, next_value, next_cost_value = self.model(preprocessed)

        for i in reversed(range(self.num_steps_per_proc)):
            next_mask = self.masks[i + 1] if i < self.num_steps_per_proc - 1 else self.mask
            next_value_i = self.values[i + 1] if i < self.num_steps_per_proc - 1 else next_value
            next_cost_value_i = self.cost_values[i + 1] if i < self.num_steps_per_proc - 1 else next_cost_value
            next_advantage = self.advantages[i + 1] if i < self.num_steps_per_proc - 1 else 0
            next_cost_advantage = self.cost_advantages[i + 1] if i < self.num_steps_per_proc - 1 else 0
            next_return = self.returnn[i + 1] if i < self.num_steps_per_proc - 1 else 0
            next_cost_return = self.cost_returnn[i + 1] if i < self.num_steps_per_proc - 1 else 0

            self.returnn[i] = self.rewards[i] + self.discount * next_return * next_mask
            self.cost_returnn[i] = torch.maximum(self.costs[i], next_cost_return * next_mask)
            delta = self.rewards[i] + self.discount * next_value_i * next_mask - self.values[i]
            cost_delta = (1 - self.cost_discount) * self.costs[i] + \
                self.cost_discount * (torch.maximum(self.costs[i], next_cost_value_i)) * next_mask - self.cost_values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask
            self.cost_advantages[i] = cost_delta + self.cost_discount * self.cost_gae_lambda * next_cost_advantage * next_mask
            self.qs[i] = self.rewards[i] + self.discount * next_value_i * next_mask
            self.cost_qs[i] = self.costs[i] + self.cost_discount * next_cost_value_i * next_mask

        exps = DictList()
        exps.obs = [
            self.obss[i][j // self.num_agents][self.agent_keys[j % self.num_agents]]
            for j in range(self.num_procs)
            for i in range(self.num_steps_per_proc)
        ]
        exps.agent_id = self.agent_ids_buf.transpose(0, 1).reshape(-1)
        exps.action = self.actions.transpose(0, 1).reshape((-1,) + self.action_space_shape)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.cost_value = self.cost_values.transpose(0, 1).reshape(-1)
        exps.qs = self.qs.transpose(0, 1).reshape(-1)
        exps.cost_qs = self.cost_qs.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.costs = self.costs.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.cost_advantage = self.cost_advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.cost_returnn = self.cost_returnn.transpose(0, 1).reshape(-1)
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        flat_agent_ids = self.agent_ids_buf.transpose(0, 1).reshape(-1).tolist()
        exps.obs = preprocess_obss_ma(exps.obs, self.propositions, flat_agent_ids, device=self.device)

        keep = max(self.log_done_counter, self.num_envs)
        logs = {
            "return_per_episode": self.log_return[-keep:],
            "cost_return_per_episode": self.log_cost_return[-keep:],
            "num_steps_per_episode": self.log_num_steps[-keep:],
            "success_per_episode": self.log_success[-keep:],
            "violation_per_episode": self.log_violation[-keep:],
            "num_steps": self.num_steps,
            "avg_goal_success": {k: float(v) / self.goal_counts[k] for k, v in self.goal_success.items()},
        }
        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_envs:]
        self.log_cost_return = self.log_cost_return[-self.num_envs:]
        self.log_num_steps = self.log_num_steps[-self.num_envs:]
        self.goal_success = defaultdict(int)
        self.goal_counts = defaultdict(int)
        return exps, logs

    @abstractmethod
    def update_parameters(self, exps):
        pass
