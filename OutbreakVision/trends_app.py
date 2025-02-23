import ssl
import certifi
import pandas as pd  # Import pandas for reading CSV
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import random
import plotly.graph_objects as go

import plotly.graph_objects as go

def generate_trend_graph(actual_x, actual_y, pred_x, pred_y, virus, country):
    """
    Generates a Plotly graph for actual vs predicted flu trends.

    :param actual_x: DataFrame containing actual case dates
    :param actual_y: DataFrame containing actual case values
    :param pred_x: DataFrame containing predicted case dates
    :param pred_y: DataFrame containing predicted case values
    :param virus: The name of the virus
    :param country: The country name
    :return: HTML representation of the graph
    """
    if actual_x is None or actual_y is None or pred_x is None or pred_y is None:
        return "<p>Error: Data could not be loaded.</p>"

    fig = go.Figure()

    # Actual Cases
    fig.add_trace(go.Scatter(
        x=actual_x['DATE'],
        y=actual_y['target_3_weeks_ahead'],
        mode='lines+markers',
        name='Actual Cases'
    ))

    # Predicted Cases
    fig.add_trace(go.Scatter(
        x=pred_x['DATE'],
        y=pred_y['target_3_weeks_ahead'],
        mode='lines+markers',
        name='Predicted Cases'
    ))

    # Update layout
    fig.update_layout(
        title=f"Predicted {virus} Cases in {country}",
        xaxis_title="Days",
        yaxis_title="Number of Cases",
        template="plotly_dark",
        width=1400,
        height=600,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
            showgrid=True,
            range=[min(pred_x['DATE']), max(pred_x['DATE'])],
        ),
        yaxis=dict(fixedrange=False),
        dragmode='zoom',
        hovermode='closest',
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Rockwell"),
        showlegend=True,
        xaxis_rangeslider_visible=True,
    )

    return fig.to_html(full_html=False)


def load_virus_data(virus: str, country: str):
    """
    Loads data for a given virus and country from the CSV files.

    :param virus: The name of the virus (e.g., 'Flu', 'COVID')
    :param country: The name of the country (e.g., 'United States')
    :return: Tuple containing actual and predicted cases data.
    """
    viruses = ['Flu', 'COVID']
    virus = 'flu'
    country = 'united_states'

    actual_x_path = f'data/actual_x_{virus}_{country}.csv'
    actual_y_path = f'data/actual_y_data_{virus}_{country}.csv'
    pred_x_path = f'data/pred_x_{virus}_{country}.csv'
    pred_y_path = f'data/pred_y_data_{virus}_{country}.csv'

    try:
        actual_x = pd.read_csv(actual_x_path)
        actual_y = pd.read_csv(actual_y_path)
        pred_x = pd.read_csv(pred_x_path)
        pred_y = pd.read_csv(pred_y_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None, None

    return actual_x, actual_y, pred_x, pred_y