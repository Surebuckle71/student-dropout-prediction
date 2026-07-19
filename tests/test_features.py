import numpy as np
import pandas as pd

from dropout_prediction.features import drop_collinear, encode_categoricals, scale_numeric


def test_encode_categoricals_aligns_val_test_columns():
    X_train = pd.DataFrame({"course": ["a", "b", "c"], "age": [18, 19, 20]})
    X_val = pd.DataFrame({"course": ["a", "a"], "age": [23, 24]})
    X_test = pd.DataFrame({"course": ["b"], "age": [25]})

    train_enc, val_enc, test_enc = encode_categoricals(
        X_train, X_val, X_test, categorical_cols=["course"]
    )

    assert list(train_enc.columns) == list(val_enc.columns) == list(test_enc.columns)
    assert "age" in train_enc.columns
    assert all(pd.api.types.is_numeric_dtype(dt) or dt == "bool" for dt in train_enc.dtypes)


def test_infer_categorical_columns_flags_low_cardinality_integer_codes():
    from dropout_prediction.features import infer_categorical_columns

    df = pd.DataFrame(
        {
            "course_code": [1, 2, 3, 1, 2] * 20,  # low cardinality -> categorical
            "grade": np.linspace(0, 20, 100),  # high cardinality -> numeric
            "label": ["a", "b"] * 50,  # object dtype -> categorical
        }
    )
    result = infer_categorical_columns(df, max_cardinality=10)

    assert "course_code" in result
    assert "label" in result
    assert "grade" not in result


def test_drop_collinear_removes_one_of_a_perfectly_correlated_pair():
    X_train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [4, 1, 3, 2]})
    X_val = X_train.copy()
    X_test = X_train.copy()

    out_train, out_val, out_test = drop_collinear(X_train, X_val, X_test, threshold=0.9)

    assert len(out_train.columns) == 2  # "a" and "b" are perfectly correlated, one is dropped
    assert "c" in out_train.columns
    assert list(out_train.columns) == list(out_val.columns) == list(out_test.columns)


def test_scale_numeric_fits_on_train_only():
    X_train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    X_val = pd.DataFrame({"x": [100.0]})
    X_test = pd.DataFrame({"x": [-100.0]})

    train_s, val_s, test_s, scaler = scale_numeric(X_train, X_val, X_test)

    assert np.isclose(train_s["x"].mean(), 0.0, atol=1e-8)
    assert scaler.mean_[0] == 2.0  # mean of the training column only
