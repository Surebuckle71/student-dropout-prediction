import numpy as np

from dropout_prediction.evaluate import compute_metrics
from dropout_prediction.models import get_candidate_models


def test_get_candidate_models_returns_ten_models():
    models = get_candidate_models()
    assert len(models) == 10


def test_all_candidate_models_fit_and_predict_proba():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = rng.integers(0, 2, size=60)

    for name, model in get_candidate_models(class_weight="balanced").items():
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (60, 2), f"{name} produced unexpected predict_proba shape"


def test_compute_metrics_keys():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_proba = [0.1, 0.8, 0.4, 0.2]

    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
