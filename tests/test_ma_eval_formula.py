"""MA deploy formula default."""
from utils.deploy_meta import MA_EVAL_FORMULA_DEFAULT


def test_ma_eval_formula_per_agent_surface():
    assert MA_EVAL_FORMULA_DEFAULT == (
        "((!surface_0 & !surface_1) U all_entrapped) & (F surface_0 & F surface_1)"
    )
