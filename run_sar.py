#!/usr/bin/env python
"""Launch SAR GenZ training across many seeds (mirrors run_zones_safety.py)."""
import os
import subprocess
import sys
from dataclasses import dataclass

import simple_parsing


@dataclass
class Args:
  script: str  # train_ppo | train_rco
  name: str
  env: str = 'PointLTL0MASAR1WC-v0'
  curriculum: str
  model_config: str
  seed: int | list[int] = 0
  num_seeds: int | None = None  # if set, runs seeds 0 .. num_seeds-1
  device: str = 'cuda:1'
  num_steps: int = 10_000_000
  num_procs: int = 24
  steps_per_process: int = 4096
  batch_size: int = 2048
  lr: float = 0.0003
  epochs: int = 10
  discount: float = 0.998
  entropy_coef: float = 0.003
  log_interval: int = 1
  save_interval: int = 10
  log_csv: bool = True
  log_wandb: bool = False
  save: bool = True
  parallel: bool = False
  vec_backend: str = 'safety_async'
  sar_env_backend: str = 'specrl'
  fast_action_bridge: bool = True


def _resolve_seeds(args: Args) -> list[int]:
  if args.num_seeds is not None:
    return list(range(args.num_seeds))
  if isinstance(args.seed, list):
    return args.seed
  return [args.seed]


def main():
  args = simple_parsing.parse(Args)
  env = os.environ.copy()
  env['PYTHONPATH'] = 'src'

  seeds = _resolve_seeds(args)
  for seed in seeds:
    command = [
      sys.executable, f'src/train/{args.script}.py',
      '--name', f'{args.name}-s{seed}',
      '--env', args.env,
      '--num_steps', str(args.num_steps),
      '--model_config', args.model_config,
      '--curriculum', args.curriculum,
      '--seed', str(seed),
      '--device', args.device,
      '--num_procs', str(args.num_procs),
      '--vec_backend', args.vec_backend,
      '--sar_env_backend', args.sar_env_backend,
      '--steps_per_process', str(args.steps_per_process),
      '--batch_size', str(args.batch_size),
      '--lr', str(args.lr),
      '--discount', str(args.discount),
      '--entropy_coef', str(args.entropy_coef),
      '--epochs', str(args.epochs),
      '--log_interval', str(args.log_interval),
      '--save_interval', str(args.save_interval),
    ]
    if args.parallel:
      command.append('--parallel')
    if args.fast_action_bridge:
      command.append('--fast_action_bridge')
    if args.log_wandb:
      command.append('--log_wandb')
    if not args.log_csv:
      command.append('--no-log_csv')
    if not args.save:
      command.append('--no-save')
    print('Running:', ' '.join(command))
    subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
  main()
