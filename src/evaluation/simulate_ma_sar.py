"""Evaluate a MASAR1WC-trained shared RCO policy on MASAR2WC with automaton coordinator."""
import argparse

from deploy.ma_rollout import EVAL_ENV, TRAIN_ENV, simulate_ma_sar
from utils.deploy_meta import MA_EVAL_FORMULA_DEFAULT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_env", type=str, default=EVAL_ENV)
    parser.add_argument("--train_env", type=str, default=TRAIN_ENV)
    parser.add_argument("--exp", type=str, default="GenZ-LTL")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--formula", type=str, default=MA_EVAL_FORMULA_DEFAULT)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-done", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--zone-compat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="48-d Zone-like SAR features for PointLtlSafety* → SAR transfer eval.",
    )
    args = parser.parse_args()
    gamma = 0.998
    simulate_ma_sar(
        args.eval_env,
        args.train_env,
        gamma,
        args.exp,
        args.seed,
        args.num_episodes,
        args.formula,
        args.render,
        args.deterministic,
        args.debug_done,
        args.device,
        zone_compat=args.zone_compat,
    )


if __name__ == "__main__":
    main()
