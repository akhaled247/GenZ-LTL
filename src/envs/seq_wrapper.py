from typing import Any, SupportsFloat, Callable

import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from ltl.automata import LDBASequence
from ltl.logic import Assignment, FrozenAssignment
from envs.env_utils import find_builder, get_env_attr

SAR_AGENT_OBS_KEYS = [
    "accelerometer_0", "velocimeter_0", "gyro_0",
    "magnetometer_0", "wall_sensor_0",
]
ZONES_SAFETY_FEAT_DIM = 48


def sar_feat_dim(lidar_bins: int, agent_obs_dim: int = 16) -> int:
    return agent_obs_dim + 3 * lidar_bins


def sar_task(env: gymnasium.Env):
    return env.unwrapped.task


def sar_agent_obs(env: gymnasium.Env, agent_idx: int = 0) -> dict:
    original_obs = sar_task(env).original_obs
    if original_obs is None:
        raise RuntimeError("task.original_obs is None — reset inner env first")
    agent_key = f"agent_{agent_idx}"
    if isinstance(original_obs, dict) and agent_key in original_obs:
        return original_obs[agent_key]
    return original_obs


def casualty_lidar_key(prop: str, agent_idx: int = 0) -> str:
    from specbench.envs.zones.sar_propositions import resolve_casualty_lidar_key
    return resolve_casualty_lidar_key(prop, agent_idx)


def lidar_keys_for_prop(prop: str, agent_idx: int = 0, num_agents: int = 1) -> list[str]:
    from specbench.envs.zones.sar_propositions import resolve_casualty_lidar_keys
    return resolve_casualty_lidar_keys(prop, agent_idx=agent_idx, num_agents=num_agents)


def buildings_lidar_key(agent_idx: int = 0) -> str:
    return f"terracotta_buildings_lidar_{agent_idx}"


def lidar_for_assignments(
    original_obs,
    assignments,
    lidar_dim: int,
    agent_idx: int = 0,
    num_agents: int = 1,
) -> np.ndarray:
    keys = []
    for assignment in assignments:
        for prop in assignment.to_string():
            keys.extend(lidar_keys_for_prop(prop, agent_idx, num_agents))
    if not keys:
        return np.zeros(lidar_dim, dtype=np.float64)
    available = [original_obs[k] for k in keys if k in original_obs]
    if not available:
        return np.zeros(lidar_dim, dtype=np.float64)
    return np.max(np.vstack(available), axis=0)


def sar_agent_obs_keys(agent_idx: int = 0) -> list[str]:
    suffix = f"_{agent_idx}"
    return [k.rsplit("_", 1)[0] + suffix for k in SAR_AGENT_OBS_KEYS]


def pre_process_obs_sar(
        env: gymnasium.Env,
        agent_obs_keys: list[str],
        reach: frozenset[FrozenAssignment],
        avoid: frozenset[FrozenAssignment],
        feat_shape: tuple[int, ...],
        agent_idx: int = 0,
) -> np.ndarray:
    original_obs = sar_agent_obs(env, agent_idx)
    lidar_dim = sar_task(env).lidar_conf.num_bins
    num_agents = getattr(sar_task(env), "agent_num", 1)
    agent_obs = np.concatenate([
        original_obs[k].flatten() if np.ndim(original_obs[k]) > 1 else original_obs[k]
        for k in agent_obs_keys
    ])
    buildings_obs = original_obs[buildings_lidar_key(agent_idx)].flatten()
    assert buildings_obs.shape == (lidar_dim,)
    reach_obs = lidar_for_assignments(
        original_obs, reach, lidar_dim, agent_idx=agent_idx, num_agents=num_agents,
    )
    avoid_obs = lidar_for_assignments(
        original_obs, avoid, lidar_dim, agent_idx=agent_idx, num_agents=num_agents,
    )
    obs = np.concatenate([agent_obs, buildings_obs, reach_obs, avoid_obs]).astype(np.float32)
    assert obs.shape == feat_shape, f"obs.shape = {obs.shape}, expected {feat_shape}"
    return obs


class SequenceWrapper(gymnasium.Wrapper):
    """
    Wrapper that adds a reach-avoid sequence of propositions to the observation space.
    """

    def __init__(self, env: gymnasium.Env, sample_sequence: Callable[[], LDBASequence], partial_reward=False):
        super().__init__(env)
        if "SAR" in env.spec.id:
            self.agent_obs_keys = SAR_AGENT_OBS_KEYS
            lidar_bins = sar_task(env).lidar_conf.num_bins
            feat_dim = sar_feat_dim(lidar_bins)
            self.observation_space = spaces.Dict({
                'features': spaces.Box(-np.inf, np.inf, (feat_dim,), dtype=np.float32),
            })
        else:
            self.observation_space = spaces.Dict({
                'features': env.observation_space,
            })
        self.unwrapped.sample_sequence = sample_sequence
        self.goal_seq = None
        self.num_reached = 0
        self.propositions = set(env.get_propositions())
        self.partial_reward = partial_reward
        self.obs = None
        self.info = None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if (action == LDBASequence.EPSILON).all():
            obs, _, terminated, truncated, info = self.apply_epsilon_action()
            reward = 0.
        else:
            assert not (action == LDBASequence.EPSILON).any()
            obs, reward, terminated, truncated, info = super().step(action)
        reach, avoid = self.goal_seq[self.num_reached]
        active_props = info['propositions']
        assignment = Assignment({p: (p in active_props) for p in self.propositions}).to_frozen()
        if assignment in avoid:
            reward = -1.
            info['violation'] = True
            terminated = True
        elif reach != LDBASequence.EPSILON and assignment in reach:
            self.num_reached += 1
            terminated = self.num_reached >= len(self.goal_seq)
            if terminated:
                info['success'] = True
            if self.partial_reward:
                reward = 1. if terminated else 1 / (len(self.goal_seq) - self.num_reached + 1)
            else:
                reward = 1. if terminated else 0
        
        reach, avoid = self.goal_seq[self.num_reached] \
            if self.num_reached < len(self.goal_seq) else self.goal_seq[-1]
        if "SAR" in self.env.spec.id:
            obs = pre_process_obs_sar(
                self.env, self.agent_obs_keys, reach, avoid,
                self.observation_space['features'].shape,
            )
        obs = self.complete_observation(obs, info)
        
        self.obs = obs
        self.info = info
        return obs, reward, terminated, truncated, info

    def apply_epsilon_action(self):
        assert self.goal_seq[self.num_reached][0] == LDBASequence.EPSILON
        self.num_reached += 1
        return self.obs, 0.0, False, False, self.info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal_seq = self.unwrapped.sample_sequence()
        self.num_reached = 0
        if "SAR" in self.env.spec.id:
            reach, avoid = self.goal_seq[self.num_reached]
            obs = pre_process_obs_sar(
                self.env, self.agent_obs_keys, reach, avoid,
                self.observation_space['features'].shape,
            )
        obs = self.complete_observation(obs, info)
        self.obs = obs
        self.info = info
        return obs, info

    def complete_observation(self, obs: WrapperObsType, info: dict[str, Any] = None) -> WrapperObsType:
        return {
            'features': obs,
            'goal': self.goal_seq[self.num_reached:],
            'initial_goal': self.goal_seq,
            'propositions': info['propositions'],
        }


class SequenceSafetyWrapper(gymnasium.Wrapper):
    """
    Wrapper that adds a reach-avoid sequence of propositions to the observation space.
    """

    def __init__(self, env: gymnasium.Env, sample_sequence: Callable[[], LDBASequence], partial_reward=False):
        super().__init__(env)
        self.region_order = get_env_attr(env, 'get_propositions')()
        if "SAR" in env.spec.id:
            self.agent_obs_keys = SAR_AGENT_OBS_KEYS
            lidar_bins = sar_task(env).lidar_conf.num_bins
            feat_dim = sar_feat_dim(lidar_bins)
            self.observation_space = spaces.Dict({
                'features': spaces.Box(-np.inf, np.inf, (feat_dim,), dtype=np.float32),
            })
        elif "PointLtlSafety" in env.spec.id:
            self.observation_space = spaces.Dict({
                'features': spaces.Box(-np.inf, np.inf, (ZONES_SAFETY_FEAT_DIM,), dtype=np.float32),
            })
            self.agent_obs_keys = ["accelerometer", "velocimeter", "gyro", "magnetometer", "wall_sensor"]
        elif "LetterSafetyEnv" in env.spec.id:
            obs_dim = env.observation_space.shape[0]
            self.observation_space = spaces.Dict({
                'features': spaces.Box(0, 1, (obs_dim, obs_dim, 1), dtype=np.float32)
            })
        self.unwrapped.sample_sequence = sample_sequence
        self.goal_seq = None
        self.num_reached = 0
        self.propositions = set(self.region_order)

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        reach, avoid = self.goal_seq[self.num_reached]
        
        active_props = info['propositions']
        assignment = Assignment({p: (p in active_props) for p in self.propositions}).to_frozen()
        
        reward = 0.0; cost = -1.0; terminated = False
        if assignment in avoid:
            cost = 1.0; info['violation'] = True
            terminated = True
        elif assignment in reach:
            reward = 1.0; info['success'] = True
            self.goal_seq = self.unwrapped.sample_sequence(assignment)
            reach, avoid = self.goal_seq[self.num_reached]
        elif 'cost_ltl_walls' in info and info['cost_ltl_walls'] > 0:
            cost = 1.0; terminated = True
        elif info.get('cost', 0) > 0:
            cost = 1.0; terminated = True

        builder = find_builder(self.env)
        if builder is not None and getattr(builder, "terminated", False):
            terminated = True
        
        obs = self.pre_process_obs(reach, avoid)
        obs = self.complete_observation_current(obs, info)
        self.obs = obs
        self.info = info
        return obs, (reward, cost), terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal_seq = self.unwrapped.sample_sequence()
        self.num_reached = 0
        reach, avoid = self.goal_seq[self.num_reached]
        
        obs = self.pre_process_obs(reach, avoid)
        obs = self.complete_observation_current(obs, info)
        return obs, info

    def pre_process_obs(self, reach, avoid):
        if "SAR" in self.env.spec.id:
            return pre_process_obs_sar(
                self.env, self.agent_obs_keys, reach, avoid,
                self.observation_space['features'].shape,
            )
        if "PointLtlSafety" in self.env.spec.id:
            return self.pre_process_obs_zones(reach, avoid)
        if "LetterSafetyEnv" in self.env.spec.id:
            return self.pre_process_obs_letter(reach, avoid)
        raise ValueError(f"Unsupported env for SequenceSafetyWrapper: {self.env.spec.id}")

    def pre_process_obs_zones(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        original_obs = self.task.original_obs
        lidar_dim = self.task.lidar_conf.num_bins
        agent_obs = np.concatenate([original_obs[key] for key in self.agent_obs_keys])

        reach_zones = [r.to_string()[0] + "_zones_lidar" for r in list(reach)]
        avoid_zones = [a.to_string()[0] + "_zones_lidar" for a in list(avoid) if a.to_string()]
        
        reach_obs = np.vstack([original_obs[color] for color in reach_zones])
        reach_obs = np.max(reach_obs, axis=0)
        if len(avoid_zones):
            avoid_obs = np.vstack([original_obs[color] for color in avoid_zones])
            avoid_obs = np.max(avoid_obs, axis=0)
        else:
            avoid_obs = np.zeros(lidar_dim)
            
        assert agent_obs.shape == reach_obs.shape == avoid_obs.shape == (lidar_dim,)
        return np.concatenate([agent_obs, reach_obs, avoid_obs])

    def pre_process_obs_letter(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        obs = self.env.original_obs
        letter_to_index = {letter: i for i, letter in enumerate(self.region_order)}
        
        reach_indices = [letter_to_index[r.to_string()[0]] for r in list(reach)]
        avoid_indices = [letter_to_index[a.to_string()[0]] for a in list(avoid)]
        
        reach_mask = np.any(obs[:, :, reach_indices] > 0, axis=2)
        avoid_mask = np.any(obs[:, :, avoid_indices] > 0, axis=2)
        agent_mask = obs[:, :, -1] > 0
        
        new_obs = np.zeros(obs.shape[:2], dtype=np.float32)
        new_obs[avoid_mask] = 0.5
        new_obs[reach_mask] = 1.0
        new_obs[agent_mask] = 0.2
        return new_obs[..., None]
        
    def complete_observation_current(self, obs: WrapperObsType, info: dict[str, Any] = None) -> WrapperObsType:
        return {
            'features': obs,
            'goal': self.goal_seq,
            'initial_goal': self.goal_seq,
            'propositions': info['propositions'],
        }
