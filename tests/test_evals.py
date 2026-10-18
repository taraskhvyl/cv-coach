from evals.run import check


def test_flags_invented_numbers():
    assert not check("Reduced costs", "Reduced costs by 40%")["no_invented_numbers"]


def test_allows_placeholders_and_original_numbers():
    assert check("Reduced costs", "Reduced costs by [X%]")["no_invented_numbers"]
    assert check("Opens up from 18% to 27%", "Raised open rate from 18% to 27%")["no_invented_numbers"]


def test_flags_weak_start_and_length():
    assert not check("x", "Responsible for the website")["strong_start"]
    assert not check("x", "Built " + "word " * 40)["concise"]
