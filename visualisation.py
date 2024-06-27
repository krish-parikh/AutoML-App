from preprocessing import *
import plotly.express as px
import plotly.graph_objects as go

def display_iqr(df, selected_col):
    fig = px.box(df, y=selected_col)
    fig.update_layout(
        title="IQR Outliers",
        xaxis_title="",
        yaxis_title=selected_col,
        )
    return fig

def display_zscore(z_scores, selected_col):
    fig = px.scatter(x=np.arange(len(z_scores)), y=z_scores, title=f'Z-Scores of {selected_col}')
    fig.add_shape(
        type="line",
        x0=0,
        x1=len(z_scores),
        y0=3,
        y1=3,
        line=dict(color="Red"),
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=len(z_scores),
        y0=-3,
        y1=-3,
        line=dict(color="Red"),
    )
    fig.update_layout(yaxis=dict(range=[min(z_scores)-1, max(z_scores)+1]))  
    return fig 

def display_categorical(df, selected_col):
    fig = px.bar(df[selected_col].value_counts().reset_index(), x='index', y=selected_col, title=f'Category frequencies in {selected_col}')
    return fig