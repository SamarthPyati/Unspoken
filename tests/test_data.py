from src.data import extract_prediction, normalize_number, is_correct


def test_boxed_preferred_over_trailing_numbers():
    assert extract_prediction("so 3+4=7, answer \\boxed{18}. Check: 2 steps") == "18"


def test_fallback_to_last_number():
    assert extract_prediction("She earns $1,234.50 in total.") == "1234.5"


def test_gold_format():
    assert normalize_number(" 72") == "72"
    assert normalize_number("1,000") == "1000"
    assert normalize_number("5.0") == "5"


def test_no_number():
    assert extract_prediction("I don't know") is None
    assert not is_correct(None, "5")


def test_equivalent_numbers_match():
    assert is_correct("18", "18.0")
    assert not is_correct("17", "18")
