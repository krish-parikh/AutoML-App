import pandas as pd
import streamlit as st
from streamlit_ydata_profiling import st_profile_report
from utilities import is_dataset_present, dataset_selection
from ydata_profiling import ProfileReport


def profile_data():
    st.title("Exploratory Data Analysis")
    st.info("This section allows you to explore your data.")
    if is_dataset_present():
        df = dataset_selection()
        profile_df = ProfileReport(df, minimal=True, orange_mode=True, explorative=True)
        st_profile_report(profile_df)
    else:
        st.error("No dataset found. Please upload a file in the 'Upload' section.")