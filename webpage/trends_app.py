import ssl
import certifi
import pandas as pd  # Import pandas for reading CSV
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import random
import plotly.graph_objects as go

# Override the default SSL context to use certifi's certificate bundle
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Initialize Flask app with explicit static folder configuration
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Define available regions and viruses
regions = {
    'United States': 331_002_651,
    'United Kingdom': 67_886_011,
    'India': 1_380_004_385,
    'Mexico': 128_932_753,
    'Russia': 145_912_025,
    'Brazil': 212_559_417,
}
viruses = ['Flu', 'COVID']
virus='flu'
country='united_states'

# Path to the webpage folder"
directory = '/Users/katieengel/Library/CloudStorage/OneDrive-Personal/Documents/Programming/OutbreakVision/Google Trends Data/webpage'

acutal_x_flu_united_states_file_path = f'{directory}/data/actual_x_{virus}_{country}.csv'
acutal_y_flu_united_states_file_path = f'{directory}/data/actual_y_data_{virus}_{country}.csv'
pred_x_flu_united_states_file_path = f'{directory}/data/pred_x_{virus}_{country}.csv'
pred_y_flu_united_states_file_path = f'{directory}/data/pred_y_data_{virus}_{country}.csv'

acutal_x_flu_united_states = pd.read_csv(acutal_x_flu_united_states_file_path)
acutal_y_flu_united_states = pd.read_csv(acutal_y_flu_united_states_file_path)
pred_x_flu_united_states = pd.read_csv(pred_x_flu_united_states_file_path)
pred_y_flu_united_states = pd.read_csv(pred_y_flu_united_states_file_path)

@app.route('/')
def index():
    return render_template('index.html', regions=regions.keys(), viruses=viruses)

@app.route('/predict', methods=['GET'])
def predict():
    virus = request.args.get('virus')
    country = request.args.get('country')
    
    if virus not in viruses or country not in regions:
        return jsonify({"error": "Invalid virus or country selection"}), 400
    
    # Create a Plotly figure
    fig = go.Figure()

    # Add line plot for actual cases
    fig.add_trace(go.Scatter(
        x=acutal_x_flu_united_states['DATE'],
        y=acutal_y_flu_united_states['target_3_weeks_ahead'],
        mode='lines+markers',
        name='Actual Cases'
    ))

    # Add line plot for predicted cases
    fig.add_trace(go.Scatter(
        x=pred_x_flu_united_states['DATE'],
        y=pred_y_flu_united_states['target_3_weeks_ahead'],
        mode='lines+markers',
        name='Predicted Cases'
    ))

    # Update layout for the plot
    fig.update_layout(
        title=f"Predicted {virus} Cases in {country}",
        xaxis_title="Days",
        yaxis_title="Number of Cases",
        template="plotly_dark",
        
        # Adjust the size of the graph (increase width and height)
        width=1400,  # Increase the width of the graph
        height=700,  # Increase the height of the graph
    
        # Set x-axis as a continuous time-based axis
        xaxis=dict(
            rangeslider=dict(visible=True),  # Adds a range slider below the graph
            type="date",  # Treat the x-axis as a date/time axis
            showgrid=True,
            range=[min(pred_x_flu_united_states['DATE']), max(pred_x_flu_united_states['DATE'])],  # Explicitly set range
        ),
        
        yaxis=dict(
            fixedrange=False  # Allow zooming on the y-axis as well
        ),
        
        # Allow zoom by dragging
        dragmode='zoom',  # Allow zoom by dragging
        hovermode='closest',
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Rockwell"),
        showlegend=True,
        
        # Enable range slider and zoom
        xaxis_rangeslider_visible=True,
    )

    # Convert the Plotly figure to HTML
    graph_html = fig.to_html(full_html=False)

    # Return the HTML page with the embedded graph
    return render_template('index.html', graph_html=graph_html, regions=regions.keys(), viruses=viruses)

if __name__ == '__main__':
    app.run(debug=True)
