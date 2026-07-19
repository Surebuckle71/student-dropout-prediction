"""Candidate model registry: 10 classifiers spanning linear, tree, margin, and instance-based methods."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Models that accept class_weight get it forwarded when handling the
# "weighted" (non-SMOTE) imbalance strategy; the rest don't support the param.
_SUPPORTS_CLASS_WEIGHT = {
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "svm",
    "decision_tree",
}


def get_candidate_models(random_state: int = 42, class_weight: str | None = None):
    weight = {} if class_weight is None else {"class_weight": class_weight}

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=random_state, **weight
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1, **weight
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1, **weight
        ),
        "adaboost": AdaBoostClassifier(random_state=random_state),
        # CalibratedClassifierCV gives SVC a predict_proba without the deprecated
        # probability=True flag (removed in sklearn 1.11).
        "svm": CalibratedClassifierCV(
            SVC(random_state=random_state, **weight), ensemble=False
        ),
        "knn": KNeighborsClassifier(),
        "decision_tree": DecisionTreeClassifier(random_state=random_state, **weight),
        "naive_bayes": GaussianNB(),
        "mlp": MLPClassifier(max_iter=500, random_state=random_state),
    }
    return models
