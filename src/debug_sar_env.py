"""Smoke probe for GenZ SAR env wiring (48-dim features, propositions)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "GenZ-LTL" / "src"), str(ROOT / "SpecRLBench")]

import specbench  # noqa: F401
from sequence.samplers import CurriculumSampler, curricula
from envs import make_env, make_env_safety

PROBES = [
    ("PointLTL0MASAR1-v0", False),
    ("PointLTL0MASAR1WC-v0", True),
]


def main():
    for env_id, safety in PROBES:
        curriculum = curricula[env_id]
        sampler = CurriculumSampler.partial(curriculum)
        factory = make_env_safety if safety else make_env
        env = factory(env_id, sampler, sequence=True, max_steps=1000)
        obs, info = env.reset(seed=0)
        assert obs["features"].shape == (48,), obs["features"].shape
        print(env_id)
        print("  propositions:", env.get_propositions())
        print("  features:", obs["features"].shape)
        print("  goal len:", len(obs["goal"]))
        print("  reset props:", info["propositions"])
        env.close()


if __name__ == "__main__":
    main()
