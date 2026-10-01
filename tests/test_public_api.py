def test_core_names_are_importable_from_the_package_root():
    from typed_decisions import Decider, ChoiceQuestion, NoulQuestion, ScoreQuestion

    assert Decider.__name__ == "Decider"
    assert ChoiceQuestion("q", ["a", "b"]).type == "choice"
    assert NoulQuestion("q").type == "noul"
    assert ScoreQuestion("q", {"a": 1.0}).type == "score"
