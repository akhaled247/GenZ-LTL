from typing import Callable

import numpy
import torch

from config import PPOConfig
from torch_ac.algos.base_ma import BaseAlgoMA


class MAIPPO(BaseAlgoMA):
    """Shared actor + per-agent critics; merged multi-agent buffer."""

    def __init__(
        self,
        envs,
        model,
        device,
        config: PPOConfig,
        preprocess_obss: Callable = None,
        fast_action_bridge: bool = False,
    ):
        del preprocess_obss
        super().__init__(
            envs,
            model,
            device,
            config.steps_per_process,
            config.discount,
            config.lr,
            config.gae_lambda,
            config.entropy_coef,
            config.value_loss_coef,
            config.max_grad_norm,
            fast_action_bridge=fast_action_bridge,
        )
        self.clip_eps = config.clip_eps
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        assert self.batch_size % self.recurrence == 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), config.lr, eps=config.optim_eps)
        self.batch_num = 0

    def update_parameters(self, exps):
        for _ in range(self.epochs):
            log_entropies, log_values, log_policy_losses, log_value_losses, log_grad_norms = [], [], [], [], []
            for inds in self._get_batches_starting_indexes():
                batch_entropy = batch_value = batch_policy_loss = batch_value_loss = 0
                batch_loss = 0
                for i in range(self.recurrence):
                    sb = exps[inds + i]
                    dist, value = self.model(sb.obs)
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    value_loss = torch.max(
                        (value - sb.returnn).pow(2),
                        (value_clipped - sb.returnn).pow(2),
                    ).mean()
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss
                batch_loss /= self.recurrence
                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in self.model.parameters() if p.requires_grad and p.grad is not None
                ) ** 0.5
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad and p.grad is not None],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                log_entropies.append(batch_entropy / self.recurrence)
                log_values.append(batch_value / self.recurrence)
                log_policy_losses.append(batch_policy_loss / self.recurrence)
                log_value_losses.append(batch_value_loss / self.recurrence)
                log_grad_norms.append(grad_norm)
        return {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
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
