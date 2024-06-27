from preprocessing import *
import plotly.express as px

def display_iqr_outliers(df, y):
    return px.box(df, y=y)

def display_zscore_outliers(z_scores, selected_columns):
    