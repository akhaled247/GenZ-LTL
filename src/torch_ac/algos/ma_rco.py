from typing import Callable

import numpy
import torch

from config import RCOConfig
from torch_ac.algos.base_ma import BaseAlgoLagMA


class MARCO(BaseAlgoLagMA):
    """Multi-agent RCO with shared actor/lagrangian and per-agent critics."""

    def __init__(
            self, envs, model, device, config: RCOConfig, preprocess_obss: Callable = None,
            fast_action_bridge: bool = False,
    ):
        del preprocess_obss
        super().__init__(
            envs, model, device, config.steps_per_process, config.discount, config.lr,
            config.gae_lambda, config.entropy_coef, config.value_loss_coef, config.max_grad_norm,
            fast_action_bridge=fast_action_bridge,
        )
        self.clip_eps = config.clip_eps
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.target_kl = config.target_kl
        self.target_cost = config.target_cost
        self.min_lag = config.min_lag
        self.max_lag = config.max_lag
        assert self.batch_size % self.recurrence == 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), config.lr, eps=config.optim_eps)
        self.batch_num = 0

    def update_parameters(self, exps):
        log_entropies, log_values, log_cost_values = [], [], []
        log_lags, log_policy_losses, log_value_losses, log_cost_value_losses = [], [], [], []
        log_lag_losses, log_grad_norms = [], []

        for n in range(self.epochs):
            iter_counts, approx_kl = 0, 0.0
            for inds in self._get_batches_starting_indexes():
                sb = exps[inds]
                dist, value, cost_value, lag = self.model(sb.obs, collect=False)
                entropy = dist.entropy().mean()
                delta_log_prob = dist.log_prob(sb.action) - sb.log_prob
                ratio = torch.exp(delta_log_prob)
                approx_kl += -delta_log_prob.mean().item()
                iter_counts += 1

                reward_surr1 = ratio * sb.advantage
                reward_surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                policy_loss_reward = torch.min(reward_surr1, reward_surr2)
                policy_loss_cost = ratio * sb.cost_advantage
                lag = torch.clamp(lag, self.min_lag, self.max_lag)
                policy_loss = (-policy_loss_reward +
                               lag.detach() * (policy_loss_cost + (1 - self.cost_discount) * sb.cost_returnn - self.target_cost)).mean()
                lag_loss = -(lag * (policy_loss_cost.detach() + (1 - self.cost_discount) * sb.cost_returnn - self.target_cost)).mean()

                value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                value_loss = torch.max(
                    (value - sb.returnn).pow(2),
                    (value_clipped - sb.returnn).pow(2),
                ).mean()
                cost_value_clipped = sb.cost_value + torch.clamp(cost_value - sb.cost_value, -self.clip_eps, self.clip_eps)
                cost_value_loss = torch.max(
                    (cost_value - sb.cost_returnn).pow(2),
                    (cost_value_clipped - sb.cost_returnn).pow(2),
                ).mean()

                total_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.value_loss_coef * cost_value_loss
                    + self.value_loss_coef * lag_loss
                    - self.entropy_coef * entropy
                )
                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in self.model.parameters() if p.requires_grad and p.grad is not None
                ) ** 0.5
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad and p.grad is not None],
                    self.max_grad_norm,
                )
                self.optimizer.step()

                log_entropies.append(entropy.item())
                log_values.append(value.mean().item())
                log_cost_values.append(cost_value.mean().item())
                log_lags.append(lag.mean().item())
                log_policy_losses.append(policy_loss.item())
                log_value_losses.append(value_loss.item())
                log_cost_value_losses.append(cost_value_loss.item())
                log_lag_losses.append(lag_loss.item())
                log_grad_norms.append(grad_norm)

            approx_kl /= iter_counts + 1e-7
            if approx_kl > self.target_kl:
                break

        return {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "cost_value": numpy.mean(log_cost_values),
            "lagrangian": numpy.mean(log_lags),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "cost_value_loss": numpy.mean(log_cost_value_losses),
            "lag_loss": numpy.mean(log_lag_losses),
            "grad_norm": numpy.mean(log_grad_norms),
        }

    def _get_batches_starting_indexes(self):
        indexes = numpy.arange(0, self.num_steps, self.recurrence)
        indexes = numpy.random.permutation(indexes)
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_steps_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1
        num_indexes = self.batch_size // self.recurrence
        return [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]
