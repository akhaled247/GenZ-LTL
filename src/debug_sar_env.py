import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # RISE-2026
sys.path[:0] = [
    str(ROOT / "GenZ-LTL" / "src"),
    str(ROOT / "SpecRLBench"),
]
import specbench  # needs pip install -e safety-gymnasium + specbench
from sequence.samplers import CurriculumSampler
from sequence.samplers.curriculum import Curriculum, EnumerateCurriculumStageZones
from envs import make_env_safety
curriculum = Curriculum([EnumerateCurriculumStageZones(threshold=None, threshold_type=None)])
sampler = CurriculumSampler.partial(curriculum)
env = make_env_safety("PointLTL0MASAR1-v0", sampler, sequence=True)
obs, info = env.reset(seed=0)
env.close()