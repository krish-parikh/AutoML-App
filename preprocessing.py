import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NAN Values

def handle_missing_values(df):
    nan_columns = df.columns[df.isna().any()].tolist()

    if not nan_columns:
        return df, None

    original_df = df.copy()

    for col in nan_columns:
        with st.expander(f"Clean {col} (missing: {df[col].isna().sum()}):"):
            if df[col].dtype in [np.number, 'float32', 'float64', 'int32', 'int64']:
                df = numeric_imputation(df, col)
            else:
                df = categorical_imputation(df, col)
    
    comparison_df = pd.concat([original_df[nan_columns], df[nan_columns]], axis=1)
    columns = [('Original', col) for col in nan_columns] + [('Cleaned', col) for col in nan_columns]
    comparison_df.columns = pd.MultiIndex.from_tuples(columns)
    return df, comparison_df

def numeric_imputation(df, col):
    impute_choice = st.selectbox(f"Choose an imputation technique for {col}:", ["Mean", "Median", "Custom"])
    apply_change = st.checkbox(f"Apply {impute_choice} to {col}")

    if apply_change:
        if impute_choice == "Mean":
            df[col].fillna(df[col].mean(), inplace=True)
        elif impute_choice == "Median":
            df[col].fillna(df[col].median(), inplace=True)
        elif impute_choice == "Custom":
            custom_value = st.number_input(f"Enter a custom value to replace NaN in {col}:")
            df[col].fillna(custom_value, inplace=True)
    return df

def categorical_imputation(df, col):
    impute_choice = st.selectbox(f"Choose an imputation technique for {col}:", ["Mode", "Custom Text"])
    apply_change = st.checkbox(f"Apply {impute_choice} to {col}")

    if apply_change:
        if impute_choice == "Mode":
            mode_value = df[col].mode().iloc[0]
            df[col].fillna(mode_value, inplace=True)
        elif impute_choice == "Custom Text":
            custom_value = st.text_input(f"Enter a custom text value to replace NaN in {col}:")
            df[col].fillna(custom_value, inplace=True)
    return df

# Column Manipulation

def drop_column(df):
    columns_to_drop = st.multiselect('Select columns to drop:', df.columns)
    df = df.drop(columns=columns_to_drop)
    return df

def drop_rows(df):
    rows_to_drop = st.multiselect('Select rows to drop:', list(df.index))
    df = df.drop(index=rows_to_drop)
    return df

# Outliers Detection

def iqr_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((df < lower_bound) | (df > upper_bound))

def zscore_outliers(df):
    z_scores = zscore(df, nan_policy='omit')
    is_outlier = np.abs(z_scores) > 3
    return is_outlier & ~np.isnan(z_scores)

def categorical_outliers(df, threshold=5):
    category_counts = df.value_counts()
    rare_categories = category_counts[category_counts < threshold].index.tolist()
    mask = df.isin(rare_categories)
    return mask, rare_categories

def handle_numerical_outliers(df, column, outliers, action, replace_with=None):
    if action == "Remove":
        return df[~outliers]
    elif action == "Replace":
        df.loc[outliers, column] = replace_with
    return df

def handle_categorical_outliers(df, column, outliers, action, replace_with="Other"):
    if action == "Remove":
        return df[~df[column].isin(outliers)]
    elif action == "Replace":
        df.loc[df[column].isin(outliers), column] = replace_with
    return df

def numeric_outlier_info(outliers, selected_col):
    if outliers.sum() > 0:
        outlier_indices_numeric = np.where(outliers)[0]
        st.write(f"Number of outliers in {selected_col}: {outliers.sum()}")
        st.write(f"Indices of outliers in {selected_col}: {', '.join(map(str, outlier_indices_numeric))}")
    else:
        st.write(f"No outliers detected in {selected_col}.")

def categorical_outlier_info(outliers, outliers_mask, selected_col):
    if outliers_mask.sum() > 0:
        outlier_indices_categorical = np.where(outliers_mask)[0]
        st.write(f"Number of outliers in {selected_col}: {outliers_mask.sum()}")
        st.write(f"Indices of outliers in {selected_col}: {', '.join(map(str, outlier_indices_categorical))}")
        st.write(f"Rare categories in {selected_col}: {', '.join(outliers)}")
    else:
        st.write(f"No outliers detected in {selected_col}.")

# Encoding

def one_hot_encode(df, selected_columns):
    df_encoded = pd.get_dummies(df, columns=selected_columns, drop_first=True)
    return df_encoded

def label_encode(df, selected_columns):
    for col in selected_columns:
        df[col], _ = pd.factorize(df[col])
    return df

# Transformations

def normalize_data(df, columns=None):
    scaler = MinMaxScaler()
    if columns:
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df, True


def standardize_data(df, columns=None):
    scaler = StandardScaler()
    if columns:
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df, True
    
