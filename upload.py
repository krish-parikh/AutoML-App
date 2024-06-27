import os
import zipfile
import streamlit as st
import pandas as pd
from utilities import load_data, is_dataset_present

def display_uploaded_dataset():
    if is_dataset_present():
        st.subheader("Uploaded Dataset")
        df = pd.read_csv('dataset.csv')
        st.dataframe(df)
        st.success("Dataset uploaded successfully!")

def handle_file_upload():
    st.subheader("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = load_data(file)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
    else: 
        st.info("No dataset uploaded. Please upload a file.")

def handle_kaggle_api_upload():
    st.subheader("Download Dataset from Kaggle")

    # Allow user to upload kaggle.json file
    kaggle_json_file = st.file_uploader("Upload your kaggle.json file:")
    if kaggle_json_file:
        kaggle_path = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_path, exist_ok=True)
        with open(f"{kaggle_path}/kaggle.json", "wb") as f:
            f.write(kaggle_json_file.read())
        os.chmod(f"{kaggle_path}/kaggle.json", 0o600)
        st.success("kaggle.json uploaded successfully!")

    api_command = st.text_input("Paste the Kaggle API command here:")
    if api_command:
        if api_command.startswith("kaggle datasets download"):
            try:
                # Execute the Kaggle API command
                download_response = os.system(api_command)
                if download_response != 0:
                    raise Exception("Failed to download the dataset.")

                # Assuming the dataset is zipped, extract it
                zip_path = api_command.split('/')[-1].split()[0] + '.zip'
                extraction_path = 'extracted_data'
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extraction_path)
                
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                                    # List all CSV files within the unzipped dataset
                csv_files = [f for f in os.listdir(extraction_path) if f.endswith('.csv')]

                # Allow the user to select the CSV file from the options
                csv_file_selected = st.selectbox("Please select the CSV file:", csv_files)
                
                if csv_file_selected:
                    df = pd.read_csv(os.path.join(extraction_path, csv_file_selected), index_col=None)
                    df.to_csv('dataset.csv', index=None)
                    st.dataframe(df)

            except Exception as e:
                st.error(f"An error occurred while processing the dataset: {str(e)}")
        else:
            st.error("Invalid Kaggle command. Please make sure it starts with 'kaggle datasets download'.")