import pandas as pd
from sklearn.preprocessing import LabelEncoder

PISTOL_ROUND_MAP = {1: 1, 13: 1}


def map_pistol_rounds(df: pd.DataFrame) -> pd.DataFrame:
    """TODO"""

    df["is_pistol_round"] = df["round"].apply(lambda x: PISTOL_ROUND_MAP.get(x, 0))
    return df


def encode_categorical_columns(
    df: pd.DataFrame, categorical_columns: list
) -> pd.DataFrame:
    """TODO"""

    for col in categorical_columns:
        df[col] = df[col].astype("category").cat.codes
    return df
