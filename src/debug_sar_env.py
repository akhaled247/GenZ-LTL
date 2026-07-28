"""Probe GenZ SequenceSafetyWrapper feature shape for SAR."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # GenZ-LTL or SpecRLBench root
sys.path.insert(0, str(ROOT / "GenZ-LTL" / "src"))   # adjust if script lives in GenZ-LTL
sys.path.insert(0, str(ROOT / "SpecRLBench"))
import specbench  # registers SAR envs
from sequence.samplers import CurriculumSampler
from sequence.samplers.curriculum import Curriculum, EnumerateCurriculumStageZones
# Until you register PointLTL0MASAR1 in curricula dict:
curriculum = Curriculum([EnumerateCurriculumStageZones(threshold=None, threshold_type=None)])
sampler = CurriculumSampler.partial(curriculum)
from envs import make_env_safety  # must resolve to your edited env_utils + seq_wrapper
env = make_env_safety("PointLTL0MASAR1-v0", sampler, sequence=True, max_steps=1000)
obs, info = env.reset(seed=0)
print("propositions:", env.get_propositions())
print("features shape:", obs["features"].shape)
print("goal:", obs["goal"])
env.close()