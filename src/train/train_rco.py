import argparse
from typing import Any, Optional

import gymnasium
import simple_parsing
import time
import datetime

import numpy as np
import torch

import config
import preprocessing
import torch_ac

import utils
from model.model import build_model_safety
from envs import make_env_safety, get_env_attr

from sequence.samplers import CurriculumSampler, curricula
from utils import torch_utils
from utils.train_device import resolve_training_device
from utils.logging.file_logger import FileLogger
from utils.logging.multi_logger import MultiLogger
from utils.logging.text_logger import TextLogger
from utils.logging.wandb_logger import WandbLogger
from utils.model_store import ModelStore
from envs.seq_wrapper import sar_task
from utils.deploy_meta import build_deploy_meta, load_deploy_meta, FEAT_RECIPE_ZONE_COMPAT
from deploy.feature_recipe import infer_feat_recipe
from config import *


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.text_logger = TextLogger(args)
        self.model_store = ModelStore.from_config(args)

    def train(self, log_csv: bool = True, log_wandb: bool = False):
        training_status, resuming = self.get_training_status()
        envs = self.make_envs(training_status["curriculum_stage"])
        if resuming:
            self.model_store.load_vocab()
        else:
            preprocessing.init_vocab(get_env_attr(envs[0], 'get_possible_assignments')())
            self.model_store.save_vocab()
        deploy_meta = load_deploy_meta(self.model_store.path) if resuming else None
        if deploy_meta is not None and deploy_meta.get("use_subgoal_one_hot") != self.args.one_hot:
            raise ValueError(
                f"--one-hot={self.args.one_hot} does not match saved deploy_meta "
                f"(use_subgoal_one_hot={deploy_meta.get('use_subgoal_one_hot')})."
            )
        if deploy_meta is not None and deploy_meta.get("entr_bldg_obs", False) != self.args.entr_bldg_obs:
            raise ValueError(
                f"--entr-bldg-obs={self.args.entr_bldg_obs} does not match saved deploy_meta "
                f"(entr_bldg_obs={deploy_meta.get('entr_bldg_obs', False)})."
            )
        saved_zone = deploy_meta.get("feat_recipe") == "zone_compat" if deploy_meta else False
        if deploy_meta is not None and saved_zone != bool(self.args.zone_compat):
            raise ValueError(
                f"--zone-compat={self.args.zone_compat} does not match saved deploy_meta "
                f"(feat_recipe={deploy_meta.get('feat_recipe')})."
            )
        num_propositions = len(get_env_attr(envs[0], 'get_propositions')())
        model = build_model_safety(
            envs[0],
            training_status,
            model_configs[self.args.model_config],
            deploy_meta=deploy_meta,
            use_subgoal_one_hot=self.args.one_hot,
            num_propositions=num_propositions,
        )
        model.to(self.args.experiment.device)
        print(model)
        async_kwargs = None
        if self.args.experiment.vec_backend == "safety_async":
            async_kwargs = self.async_factory_kwargs(training_status["curriculum_stage"])
        algo = torch_ac.RCO(
            envs, model, self.args.experiment.device, self.args.rco,
            preprocess_obss=preprocessing.preprocess_obss,
            parallel=self.args.experiment.parallel,
            vec_backend=self.args.experiment.vec_backend,
            fast_action_bridge=self.args.experiment.fast_action_bridge,
            async_factory_kwargs=async_kwargs,
        )
        if "optimizer_state" in training_status:
            algo.optimizer.load_state_dict(training_status["optimizer_state"])
            self.text_logger.info("Loaded optimizer from existing run.")
        logger = self.make_logger(log_csv, log_wandb, resuming)
        logger.log_config()

        self.text_logger.info(f'Num parameters: {torch_utils.get_number_of_params(model)}')
        num_steps = training_status["num_steps"]
        num_updates = training_status["num_updates"]
        num_eval_steps = training_status["num_eval_steps"]
        while num_steps < self.args.experiment.num_steps:
            if self.args.save and (num_updates == 0 or num_eval_steps >= self.args.experiment.eval_interval):
                num_eval_steps = 0
                training_status = {"num_steps": num_steps, "num_updates": num_updates,
                                   "model_state": algo.model.state_dict()}
                self.model_store.save_eval_training_status(training_status)
            start = time.time()
            exps, logs = algo.collect_experiences()
            curriculum = get_env_attr(envs[0], 'sample_sequence').curriculum
            success_episodes = logs.get('success_per_episode', [])
            episode_success_rate = (
                float(np.mean(success_episodes)) if success_episodes else None
            )
            curriculum.update_task_success(
                logs['avg_goal_success'],
                episode_success_rate=episode_success_rate,
                verbose=True,
            )
            update_logs = algo.update_parameters(exps)
            logs.update(update_logs)
            update_time = time.time() - start

            num_steps += logs["num_steps"]
            num_eval_steps += logs["num_steps"]
            num_updates += 1
            if num_updates % self.args.experiment.log_interval == 0 or curriculum.finished:
                logs = self.augment_logs(logs, update_time, num_steps)
                logger.log(logs)
            if (
                    curriculum.finished or self.args.experiment.save_interval > 0 and num_updates % self.args.experiment.save_interval == 0) \
                    and self.args.save:
                training_status = {"num_steps": num_steps, "num_updates": num_updates,
                                   "model_state": algo.model.state_dict(),
                                   "optimizer_state": algo.optimizer.state_dict(),
                                   "curriculum_stage": curriculum.stage_index,
                                   "num_eval_steps": num_eval_steps,
                                   }
                self.model_store.save_training_status(training_status)
                self.write_deploy_meta(algo.model, envs[0])
                self.text_logger.info("Saved training status")
            if curriculum.finished:
                self.text_logger.important_info("Finished curriculum.")
                break

    def write_deploy_meta(self, model, env) -> None:
        actor_input_dim = int(model.actor.enc[0].in_features)
        use_env_net = model.env_net is not None
        num_props = len(get_env_attr(env, 'get_propositions')())
        subgoal_dim = 2 * num_props if getattr(model, "use_subgoal_one_hot", False) else 0
        raw_feature_dim = (
            int(model.env_net.mlp[0].in_features) if use_env_net
            else actor_input_dim - subgoal_dim
        )
        lidar_bins = sar_task(env).lidar_conf.num_bins
        recipe = (
            FEAT_RECIPE_ZONE_COMPAT if self.args.zone_compat
            else infer_feat_recipe(raw_feature_dim, lidar_bins)
        )
        meta = build_deploy_meta(
            train_env=self.args.experiment.env,
            raw_feature_dim=raw_feature_dim,
            use_env_net=use_env_net,
            actor_input_dim=actor_input_dim,
            feat_recipe=recipe,
            use_subgoal_one_hot=self.args.one_hot,
            entr_bldg_obs=False if self.args.zone_compat else self.args.entr_bldg_obs,
            num_propositions=num_props,
        )
        path = self.model_store.save_deploy_meta(meta)
        self.text_logger.info(f"Wrote deploy metadata to {path}")

    def make_probe_env(self, curriculum_stage: int) -> gymnasium.Env:
        curriculum = curricula[self.args.curriculum]
        curriculum.stage_index = curriculum_stage
        self.text_logger.important_info(f"Curriculum stage: {curriculum.stage_index}")
        sampler = CurriculumSampler.partial(curriculum)
        return make_env_safety(
            self.args.experiment.env,
            sampler,
            sequence=True,
            sar_env_backend=self.args.experiment.sar_env_backend,
            max_steps=2500,
            entr_bldg_obs=False if self.args.zone_compat else self.args.entr_bldg_obs,
            zone_compat=self.args.zone_compat,
        )

    def async_factory_kwargs(self, curriculum_stage: int) -> dict[str, Any]:
        return {
            "n_envs": self.args.experiment.num_procs,
            "env_name": self.args.experiment.env,
            "curriculum_name": self.args.curriculum,
            "curriculum_stage": curriculum_stage,
            "seed": self.args.experiment.seed,
            "max_steps": 2500,
            "sar_env_backend": self.args.experiment.sar_env_backend,
            "safety": True,
            "sequence": True,
            "entr_bldg_obs": False if self.args.zone_compat else self.args.entr_bldg_obs,
            "zone_compat": self.args.zone_compat,
        }

    def make_envs(self, curriculum_stage: int) -> list[gymnasium.Env]:
        utils.set_seed(self.args.experiment.seed)
        if self.args.experiment.vec_backend == "safety_async":
            env = self.make_probe_env(curriculum_stage)
            seed_offset = 100 * self.args.experiment.seed
            env.reset(seed=seed_offset)
            self.text_logger.info(
                f"Async vec backend: probe env on main; {self.args.experiment.num_procs} workers in subprocesses."
            )
            return [env]

        envs = []
        for i in range(self.args.experiment.num_procs):
            envs.append(self.make_probe_env(curriculum_stage))
        seed_offset = 100 * self.args.experiment.seed
        seeds = [seed_offset + i for i in range(self.args.experiment.num_procs)]
        self.text_logger.info(f"Using seeds: {seeds}")
        for env, seed in zip(envs, seeds):
            env.reset(seed=seed)
        self.text_logger.info("Environments loaded.")
        return envs

    def get_training_status(self) -> tuple[dict, bool]:
        resuming = False
        try:
            training_status = self.model_store.load_training_status(map_location='cpu')
            self.text_logger.important_info("Resuming training from existing run.")
            resuming = True
        except FileNotFoundError:
            training_status = {"num_steps": 0, "num_updates": 0, "curriculum_stage": 0, "num_eval_steps": 0}
        return training_status, resuming

    def load_pretrained_model(self) -> Optional[dict]:
        if self.args.pretraining_experiment is not None:
            pretrained = self.model_store.load_pretrained()
            self.text_logger.important_info(f"Loaded pretrained model.")
            return pretrained
        return None

    def make_logger(self, log_csv: bool, log_wandb: bool, resuming: bool) -> MultiLogger:
        loggers = [self.text_logger]
        if log_csv:
            loggers.append(FileLogger(self.args, resuming=resuming))
        if log_wandb:
            loggers.append(WandbLogger(self.args, project_name='deep-ltl', resuming=resuming))
        return MultiLogger(*loggers)

    def augment_logs(self, logs: dict, update_time: float, num_steps: int) -> dict:
        sps = logs["num_steps"] / update_time
        remaining_duration = int((self.args.experiment.num_steps - num_steps) / sps)
        remaining_duration = 0 if remaining_duration < 0 else remaining_duration
        remaining_time = str(datetime.timedelta(seconds=remaining_duration))

        average_reward_per_step = utils.average_reward_per_step(logs["return_per_episode"],
                                                                logs["num_steps_per_episode"])
        average_discounted_return = utils.average_discounted_return(logs["return_per_episode"],
                                                                    logs["num_steps_per_episode"],
                                                                    self.args.rco.discount)
        logs.update({
            "arps": average_reward_per_step,
            "adr": average_discounted_return,
            'sps': sps,
            'remaining': remaining_time,
            'num_steps': num_steps  # set num_steps to the total number of steps
        })
        return logs


# noinspection PyTypeChecker
def parse_arguments() -> argparse.Namespace:
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(config.ExperimentConfig, dest="experiment")
    parser.add_arguments(config.RCOConfig, dest="rco")
    parser.add_argument("--model_config", type=str, default="default", choices=model_configs.keys(),
                        required=True)
    parser.add_argument("--curriculum", type=str, choices=curricula.keys(), required=True)
    parser.add_argument("--pretraining_experiment", type=str, default=None)
    parser.add_argument("--freeze_pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log_csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--one-hot',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Concat current reach/avoid one-hot to env_net embedding (RCO subgoal conditioning).',
    )
    parser.add_argument(
        '--cost-clipping',
        action=argparse.BooleanOptionalAction,
        default=False,
        dest='cost_clipping',
        help='Clip cost policy surrogate like reward (PPO trust region on cost advantage).',
    )
    parser.add_argument(
        '--entr-bldg-obs',
        action=argparse.BooleanOptionalAction,
        default=False,
        dest='entr_bldg_obs',
        help='Max-pool building lidar into entrapped reach subgoal lidar slice.',
    )
    parser.add_argument(
        '--zone-compat',
        action=argparse.BooleanOptionalAction,
        default=False,
        dest='zone_compat',
        help='48-d zones-like SAR features (agent|reach|avoid); always max(building,entrapped); '
             'use with walls-always-in-avoid curriculum on L1 WC.',
    )
    args = parser.parse_args()
    args.rco.cost_clipping = args.cost_clipping

    args.experiment.device = resolve_training_device(args.experiment.device)

    if args.pretraining_experiment is None and args.freeze_pretrained:
        raise ValueError("Cannot freeze without providing a pretrained model.")
    if args.zone_compat and args.entr_bldg_obs:
        raise ValueError("--zone-compat already pools buildings into entrapped; omit --entr-bldg-obs.")
    return args


def main():
    args = parse_arguments()
    trainer = Trainer(args)
    start_time = time.time()
    trainer.train(log_csv=args.log_csv, log_wandb=args.log_wandb)
    training_time = datetime.timedelta(seconds=int(time.time() - start_time))
    print(f"Training took {training_time}.")


if __name__ == '__main__':
    main()
