import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(df, selected_columns):
    df_encoded = pd.get_dummies(df, columns=selected_columns, drop_first=True)
    return df_encoded

def label_encode(df, selected_columns):
    for col in selected_columns:
        df[col], _ = pd.factorize(df[col])
    return df