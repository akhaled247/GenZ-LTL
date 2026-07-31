"""Build env + model + Büchi search for SAR LTL eval (PPO or RCO)."""
from __future__ import annotations

from config import model_configs
from deploy.checkpoint_kind import detect_rl_algo, ppo_model_config_key
from deploy.loading import load_model_for_deploy
from deploy.ppo_loading import attach_ppo_sar_fields
from envs import make_env, make_env_safety
from envs.env_utils import is_safety_model_env
from ltl import FixedSampler
from model.model import build_model, build_model_safety
from sequence.search import ExhaustiveSearch, ExhaustiveSearchSafety
from utils.model_store import ModelStore
from utils.deploy_meta import load_deploy_meta


def build_sar_ltl_eval_stack(
    train_env: str,
    exp: str,
    seed: int,
    formula: str,
    *,
    eval_env: str | None = None,
    flat: bool = True,
    render_mode: str | None = None,
    ma_deploy: bool = False,
):
    """Return (env, model, search, propositions, algo)."""
    eval_env = eval_env or train_env
    if not is_safety_model_env(train_env) and "MASAR" not in train_env:
        raise ValueError(f"Not a SAR safety env: {train_env}")

    sampler = FixedSampler.partial(formula)
    model_store = ModelStore(train_env, exp, seed, None)
    training_status = model_store.load_training_status(map_location="cpu")
    algo = detect_rl_algo(training_status["model_state"])

    if algo == "ppo":
        model_store.load_vocab()
        if ma_deploy:
            env = make_env(
                eval_env, sampler, flat=False, sequence=False,
                render_mode=render_mode, max_steps=2500,
            )
        else:
            env = make_env(
                eval_env, sampler, flat=flat, sequence=False,
                render_mode=render_mode, max_steps=2500,
            )
        probe = make_env(train_env, sampler, flat=True, sequence=True, max_steps=2500)
        try:
            config_key = ppo_model_config_key(train_env)
            model = build_model(probe, training_status, model_configs[config_key])
            attach_ppo_sar_fields(model, probe)
        finally:
            probe.close()
        props = env.get_propositions()
        search = ExhaustiveSearchSafety(env, model, props, num_loops=2)
    else:
        config = model_configs[train_env]
        deploy_meta = load_deploy_meta(model_store.path)
        if ma_deploy:
            model, _meta = load_model_for_deploy(train_env, exp, seed, formula)
            env = make_env_safety(
                eval_env, sampler, flat=False, sequence=False,
                render_mode=render_mode, max_steps=2500,
                entr_bldg_obs=bool(deploy_meta.get("entr_bldg_obs", False)),
            )
        else:
            env = make_env_safety(
                eval_env, sampler, flat=flat, sequence=False,
                render_mode=render_mode, max_steps=2500,
                entr_bldg_obs=bool(deploy_meta.get("entr_bldg_obs", False)),
            )
            model = build_model_safety(
                env, training_status, config, deploy_meta=deploy_meta,
            )
        props = env.get_propositions()
        search = ExhaustiveSearchSafety(env, model, props, num_loops=2)

    return env, model, search, props, algo
