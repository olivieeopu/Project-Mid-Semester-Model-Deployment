import pandas as pd

def load_data(feature_path, target_path):
    X = pd.read_csv(feature_path)
    y = pd.read_csv(target_path)

    df = X.merge(y, on="Student_ID")
    return df


def split_regression(df):
    df = df.drop(columns=["Student_ID", "placement_status"], errors='ignore')
    X = df.drop(columns=["salary_lpa"])
    y = df["salary_lpa"]
    return X, y


def split_classification(df):
    df = df.drop(columns=["Student_ID", "salary_lpa"], errors='ignore')
    X = df.drop(columns=["placement_status"])
    y = df["placement_status"]
    return X, y