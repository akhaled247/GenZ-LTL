"""Smoke probe for GenZ SAR env wiring (48-dim features, propositions)."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "GenZ-LTL" / "src"), str(ROOT / "SpecRLBench")]

import specbench  # noqa: F401
from sequence.samplers import CurriculumSampler, curricula
from envs import make_env, make_env_safety
from envs.sar_factory import make_sar_base_env

PROBES = [
    ("PointLTL0MASAR1-v0", False),
    ("PointLTL0MASAR1WC-v0", True),
]


def _benchmark_reset(env_id: str, backend: str) -> None:
    env = make_sar_base_env(env_id, backend=backend)
    t0 = time.perf_counter()
    env.reset(seed=0)
    first = time.perf_counter() - t0
    t1 = time.perf_counter()
    env.reset(seed=1)
    second = time.perf_counter() - t1
    env.close()
    print(f"  reset benchmark ({backend}): first={first:.3f}s second={second:.3f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-reset", action="store_true")
    parser.add_argument("--sar-env-backend", default="specrl", choices=["specrl", "genz_local"])
    cli = parser.parse_args()

    if cli.benchmark_reset:
        for env_id, _ in PROBES:
            print(env_id)
            _benchmark_reset(env_id, cli.sar_env_backend)
        return

    for env_id, safety in PROBES:
        curriculum = curricula[env_id]
        sampler = CurriculumSampler.partial(curriculum)
        factory = make_env_safety if safety else make_env
        env = factory(
            env_id, sampler, sequence=True, max_steps=1000,
            sar_env_backend=cli.sar_env_backend,
        )
        obs = env.reset(seed=0)
        assert obs["features"].shape == (48,), obs["features"].shape
        print(env_id)
        print("  propositions:", env.get_propositions())
        print("  features:", obs["features"].shape)
        print("  goal len:", len(obs["goal"]))
        print("  reset props:", obs["propositions"])
        env.close()


if __name__ == "__main__":
    main()
