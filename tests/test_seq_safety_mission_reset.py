"""SequenceSafetyWrapper must surface builder termination for vec auto-reset."""
import numpy as np

from envs.env_utils import find_builder, make_env_safety
from sequence.samplers import CurriculumSampler, curricula


def _make_stack():
    curriculum = curricula["PointLTL0MASAR1WC-v0"]
    props = ["entrapped_0", "surface_0"]
    sampler = CurriculumSampler.partial(curriculum)(props)
    return make_env_safety(
        "PointLTL0MASAR1WC-v0",
        sampler,
        sequence=True,
        max_steps=2500,
    )


def test_mission_complete_yields_done_and_allows_next_step():
    env = _make_stack()
    try:
        env.reset(seed=0)
        builder = find_builder(env)
        assert builder is not None
        builder.task.goal_achieved = [True] * builder.task.agent_num

        _obs, _reward, done, _info = env.step(env.action_space.sample())
        assert done, "builder mission complete must propagate as done for worker reset"

        env.reset(seed=1)
        env.step(env.action_space.sample())
    finally:
        env.close()


def test_subgoal_success_without_builder_done_continues_episode():
    env = _make_stack()
    try:
        env.reset(seed=2)
        builder = find_builder(env)
        assert builder is not None
        assert not builder.done

        _obs, _reward, done, _info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert not done or builder.done
    finally:
        env.close()
