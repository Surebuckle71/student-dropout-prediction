from dropout_prediction.predict import DropoutPredictor


def test_predictor_loads_artifact_and_returns_probability(trained_artifact, cleaned_df):
    predictor = DropoutPredictor(str(trained_artifact))
    sample = cleaned_df.drop(columns=["target_binary"]).iloc[0].to_dict()

    result = predictor.predict(sample)

    assert 0.0 <= result["dropout_probability"] <= 1.0
    assert isinstance(result["at_risk"], bool)
    assert result["model_name"] == "logistic_regression"


def test_predictor_handles_missing_and_unknown_fields(trained_artifact):
    predictor = DropoutPredictor(str(trained_artifact))
    result = predictor.predict({"age_at_enrollment": 20, "some_unknown_field": "x"})

    assert 0.0 <= result["dropout_probability"] <= 1.0
