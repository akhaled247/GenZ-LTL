"""MA deploy formula default."""
from utils.deploy_meta import MA_EVAL_FORMULA_DEFAULT, warn_if_fragile_ma_formula


def test_ma_eval_formula_per_agent_surface():
    assert MA_EVAL_FORMULA_DEFAULT == (
        "(!walls U ((!surface_0 & !surface_1) U all_entrapped)) & "
        "(!walls U (F surface_0 & F surface_1))"
    )


def test_warn_if_fragile_ma_formula_detects_per_agent_until(capsys):
    warn_if_fragile_ma_formula(
        "((!surface_0 U entrapped_0) & F surface_0) & ((!surface_1 U entrapped_1) & F surface_1)"
    )
    err = capsys.readouterr().out
    assert "WARNING" in err
    assert "all_entrapped" in err


def test_warn_if_fragile_ma_formula_silent_on_canonical(capsys):
    warn_if_fragile_ma_formula(MA_EVAL_FORMULA_DEFAULT)
    assert capsys.readouterr().out == ""
