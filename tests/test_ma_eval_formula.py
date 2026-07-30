"""MA deploy formula default."""
from utils.deploy_meta import MA_EVAL_FORMULA_DEFAULT


def test_ma_eval_formula_team_entrapped_gate():
    assert MA_EVAL_FORMULA_DEFAULT == (
        "((!surface_0 & !surface_1 & !all_surface) U all_entrapped) & F all_surface"
    )
