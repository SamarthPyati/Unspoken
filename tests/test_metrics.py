from src.metrics import bootstrap_ci, mcnemar, paired_diff_ci, relative_reduction


def test_bootstrap_contains_mean():
    ci = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 30)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert abs(ci["mean"] - 0.7) < 1e-9


def test_mcnemar_identical_methods():
    r = mcnemar([True, False, True], [True, False, True])
    assert r["p_value"] == 1.0 and r["only_a_correct"] == 0


def test_mcnemar_clear_difference():
    a = [True] * 30 + [False] * 70
    b = [False] * 100
    assert mcnemar(a, b)["p_value"] < 0.001


def test_paired_diff_and_reduction():
    assert paired_diff_ci([1, 1, 1], [0, 0, 0])["mean"] == 1.0
    assert abs(relative_reduction([100, 100], [40, 40]) - 0.6) < 1e-9
