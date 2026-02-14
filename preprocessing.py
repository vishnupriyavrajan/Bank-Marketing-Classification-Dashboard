import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET = "y"

def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Binary target encoding
    df[TARGET] = df[TARGET].map({"yes": 1, "no": 0})

    # Encode categorical features
    cat_cols = df.select_dtypes(include="object").columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    return train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
