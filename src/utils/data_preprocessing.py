import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

PISTOL_ROUND_MAP = {1: 1, 13: 1}


def map_pistol_rounds(df: pd.DataFrame) -> pd.DataFrame:
    """TODO"""

    df["is_pistol_round"] = df["round"].apply(lambda x: PISTOL_ROUND_MAP.get(x, 0))
    return df


def encode_categorical_columns(
    df: pd.DataFrame, categorical_columns: list
) -> pd.DataFrame:
    """TODO"""
    label_encoders = {}
    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        label_encoders[column] = encoder
    return df, label_encoders
