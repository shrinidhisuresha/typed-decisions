def test_calibration_names_are_importable_from_the_package_root():
    from typed_decisions import Calibration, Prediction, report, fit_temperature_from_predictions

    assert Calibration(permutations=3).permutations == 3
    assert Prediction({"a": 1.0}, "a").label == "a"
    assert callable(report)
    assert callable(fit_temperature_from_predictions)
