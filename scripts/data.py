import pandas as pd

KC_COLS = ["kc1_sample_space", "kc2_conditioning", "kc3_joint_chain",
           "kc4_total_probability", "kc5_bayes_update"]
FLAG_COLS = ["conjunction", "inverse", "time_axis",
             "denominator_neglect", "base_rate_neglect"]


def _split_kcs(cell):
    return cell.split(";") if isinstance(cell, str) and cell else []


def load_data(path):
    """Load the annotated csv.

    Converts designed_kcs and adaptive_kcs into python lists in-column,
    drops the Q0 warm-up rows, and returns the dataframe. Nothing else.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    df["designed_kcs"] = df["designed_kcs"].apply(_split_kcs)
    df["adaptive_kcs"] = df["adaptive_kcs"].apply(_split_kcs)
    df["question_number"] = df["question_number"].astype(int)
    df = df[df["question_number"] != 0].reset_index(drop=True)
    return df
