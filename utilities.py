import pandas as pd
import os
import streamlit as st
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file):
    return pd.read_csv(file, index_col=None)

def is_dataset_present():
    return os.path.exists('./dataset.csv')

def is_new_dataset_present():
    return os.path.exists('./dataset_updated.csv')

def delete_all_uploaded_files():
    try:
        # Delete the trained model
        if os.path.exists('trained_model.h5'):
            os.remove('trained_model.h5')
                       
        # Delete the dataset.csv file
        if os.path.exists('dataset.csv'):
            os.remove('dataset.csv')

        # Delete the dataset_backup.csv file
        if os.path.exists('dataset_updated.csv'):
            os.remove('dataset_updated.csv')
            
        # Delete the Kaggle JSON file if you have it stored
        kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
        if os.path.exists(kaggle_path):
            os.remove(kaggle_path)

        # Delete unzipped files
        extraction_path = 'extracted_data'
        if os.path.exists(extraction_path):
            shutil.rmtree(extraction_path)
        
        # Delete downloaded ZIP files
        for file_name in os.listdir():
            if file_name.endswith('.zip'):
                os.remove(file_name)

        # Clear cache
        st.cache_data.clear()

        st.success("All uploaded files have been deleted!")
    except Exception as e:
        st.error(f"An error occurred while deleting the files: {str(e)}")

def dataset_selection():
    if is_new_dataset_present():
        return load_data('dataset_updated.csv')
    elif is_dataset_present():
        return load_data('dataset.csv')

def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def string_columns(df):
    return df.select_dtypes(exclude=[np.number]).columns.tolist()

def categorical_columns(df):
    return df.select_dtypes(['object']).columns.tolist()

def save_dataset(df):
    if st.button("Save Changes"):
        df.to_csv('dataset_updated.csv', index=None)
        st.success("Your dataset has been updated!")