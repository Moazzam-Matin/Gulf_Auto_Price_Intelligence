import sys
sys.path.insert(0, ".")

import pandas as pd
from src.preprocess import fit_target_encoding, apply_target_encoding


def test_fit_target_encoding_only_uses_given_rows():
    """
    The encoding mapping must be computed ONLY from the rows passed in.
    This is the exact bug we fixed in train.py - encoding was being
    calculated on the full dataset (train + test) before the split.
    """
    train_df = pd.DataFrame({
        "Make": ["toyota", "toyota", "kia"],
        "Log_Price": [10.0, 10.0, 8.0],
    })

    mappings = fit_target_encoding(train_df, categorical_cols=["Make"], alpha=10)

    # The mapping should only contain categories seen in train_df
    assert "toyota" in mappings["Make"].index
    assert "kia" in mappings["Make"].index
    assert "nissan" not in mappings["Make"].index

def test_encoding_is_insensitive_to_test_row_values():
    """
    The actual leakage guarantee: changing what's in the test set's
    Log_Price must NEVER change the encoded value it receives, because
    the mapping is fixed at fit-time and only ever looked up, not recomputed.
    """
    train_df = pd.DataFrame({
        "Make": ["toyota", "toyota", "kia", "kia"],
        "Log_Price": [10.0, 10.0, 8.0, 8.0],
    })

    mappings = fit_target_encoding(train_df, categorical_cols=["Make"], alpha=10)

    test_df_a = pd.DataFrame({"Make": ["toyota"], "Log_Price": [2.0]})
    test_df_b = pd.DataFrame({"Make": ["toyota"], "Log_Price": [999.0]})

    encoded_a = apply_target_encoding(test_df_a, ["Make"], mappings)["Make_Encoded_Log"].iloc[0]
    encoded_b = apply_target_encoding(test_df_b, ["Make"], mappings)["Make_Encoded_Log"].iloc[0]

    assert encoded_a == encoded_b, (
        "Encoded value changed based on the test row's own price - this means "
        "the encoding is leaking test-set information!"
    )