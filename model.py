import pandas as pd

def feature_target_split(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y