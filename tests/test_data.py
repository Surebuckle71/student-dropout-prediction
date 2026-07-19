import pandas as pd

from dropout_prediction.data import binarize_target, train_val_test_split


def test_binarize_target_drops_enrolled_and_maps_labels():
    df = pd.DataFrame({"target": ["Dropout", "Graduate", "Enrolled", "Dropout"]})
    out = binarize_target(df)

    assert "target" not in out.columns
    assert len(out) == 3  # the "Enrolled" row is dropped
    assert set(out["target_binary"].unique()) <= {0, 1}
    assert out["target_binary"].sum() == 2  # two Dropout rows


def test_train_val_test_split_is_60_20_20(cleaned_df):
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(cleaned_df)
    n = len(cleaned_df)

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_val) + len(X_test) == n
    assert abs(len(X_train) / n - 0.6) < 0.02
    assert abs(len(X_val) / n - 0.2) < 0.02
    assert abs(len(X_test) / n - 0.2) < 0.02


def test_split_has_no_row_overlap(cleaned_df):
    X_train, X_val, X_test, *_ = train_val_test_split(cleaned_df)
    train_idx, val_idx, test_idx = set(X_train.index), set(X_val.index), set(X_test.index)

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
