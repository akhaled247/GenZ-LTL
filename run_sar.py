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
  env: str
  curriculum: str
  model_config: str
  seed: int | list[int] = 0
  num_seeds: int | None = None  # if set, runs seeds 0 .. num_seeds-1
  device: str = 'cuda:1'
  num_steps: int = 5_000_000
  num_procs: int = 16
  log_csv: bool = True
  log_wandb: bool = False
  save: bool = True
  parallel: bool = False
  vec_backend: str = 'list'
  sar_env_backend: str = 'specrl'
  fast_action_bridge: bool = False


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

  if args.script == 'train_ppo':
    if args.env.endswith('WC-v0'):
      raise ValueError('Use train_rco for WC / safety SAR envs')
    if args.curriculum != args.env:
      print('Warning: curriculum differs from env id', args.curriculum, args.env)
  elif args.script == 'train_rco':
    if not args.env.endswith('WC-v0'):
      print('Warning: train_rco is intended for WC safety SAR envs')
  else:
    raise ValueError(f'Unknown script: {args.script}')

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
