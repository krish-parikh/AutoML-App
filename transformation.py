import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import streamlit as st

def normalize_data(df, columns=None):
    scaler = MinMaxScaler()
    try:
        if columns:
            df[columns] = scaler.fit_transform(df[columns])
        else:
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df, True
    except ValueError:
        st.error("Your data seems to contain categorical data. Please specify columns without categorical data or choose a different preprocessing technique.")
        return df, False

def standardize_data(df, columns=None):
    scaler = StandardScaler()
    try:
        if columns:
            df[columns] = scaler.fit_transform(df[columns])
        else:
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df, True
    except ValueError:
        st.error("Your data seems to contain categorical data. Please specify columns without categorical data or choose a different preprocessing technique.")
        return df, False
