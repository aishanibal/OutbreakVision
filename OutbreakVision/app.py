from flask import Flask, request, jsonify, render_template
from pathlib import Path
import csv
import pandas as pd
from seir_model import process_country_data
from random_forest import train_model, predict_impact_score  # Import functions from the separate file
from impact_images_and_simulation import generate_impact_video
from impact_score_generator import generate_impact_scores
from trends_app import generate_trend_graph, load_virus_data
import joblib
import numpy as np
import os
# Initialize Flask app with explicit static folder configuration
app = Flask(__name__, static_folder='static', static_url_path='/static')


@app.route('/map')
# def index():
#     """
#     Render the homepage with CSV data.
#     """
#     train_model()
#
#     # Example user inputs (replace with actual user inputs from the frontend)
#     example_inputs = [46, 50, 50000, 95, 0.2, 70, 100]  # Replace with actual feature values
#     predicted_score = predict_impact_score(example_inputs)
#     print(f"Predicted Impact Score: {predicted_score:.2f}")
#     try:
#         csv_data = read_csv("data/combined_country_data.csv")
#         return render_template('index.html', data=csv_data)
#     except Exception as e:
#         return render_template('error.html', error_message=f"Error reading CSV file: {str(e)}")
@app.route('/map', methods=['GET', 'POST'])
def map():
    """
    Render the map visualization page.
    """
    if request.method == 'GET':
        return render_template('map.html', video_filename=None, snapshots=[])

    elif request.method == 'POST':
        incubation_period = float(request.form.get("incubation_period", 14))  # Default to 14 days
        decay_factor = 1 / incubation_period  # Compute decay factor

        # Step 1: Generate impact scores using decay_factor
        generate_impact_scores(decay_factor)  # This updates "impact_overtime.csv"

        # Step 2: Generate the map visualization (which reads from the updated CSV)
        video_filename = generate_impact_video()  # No parameters needed, just reads the CSV

        snapshot_folder = "ImpactScore_snapshots"

        # Get the list of snapshot files
        if os.path.exists(snapshot_folder):
            snapshots = sorted([f for f in os.listdir(snapshot_folder) if f.endswith('.png')])
        else:
            snapshots = []

        return render_template('map.html', video_filename=video_filename, snapshots=snapshots)


@app.route('/')
def home():
    return render_template("index.html")

# Route to handle predictions
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Render the prediction page and handle predictions.
    """
    if request.method == 'GET':
        # Render the prediction form
        return render_template('predict.html')
    elif request.method == 'POST':
        try:
            # Get user inputs from the request
            data = request.json
            user_inputs = data["inputs"]

            # Validate inputs
            if len(user_inputs) != 7:
                raise ValueError("Expected 7 input values")

            # Make prediction (replace with your prediction logic)
            predicted_impact = predict_impact_score(user_inputs)

            # Ensure prediction is within bounds
            predicted_impact = float(np.clip(predicted_impact, 1, 100))

            # Return the result as JSON
            return jsonify({
                "score": predicted_impact,
                "success": True
            })
        except Exception as e:
            return jsonify({
                "error": str(e),
                "success": False
            }), 400

# Available Regions and Viruses
regions = {
    'United States': 331_002_651,
    'United Kingdom': 67_886_011,
    'India': 1_380_004_385,
    'Mexico': 128_932_753,
    'Russia': 145_912_025,
    'Brazil': 212_559_417,
}
viruses = ['Flu', 'COVID']

@app.route('/select')
def select():
    return render_template('trends.html', regions=regions.keys(), viruses=viruses)

@app.route('/trends', methods=['GET'])
def trends():
    virus = request.args.get('virus')
    country = request.args.get('country')

    if virus not in viruses or country not in regions:
        return jsonify({"error": "Invalid virus or country selection"}), 400

    # Load the data
    actual_x, actual_y, pred_x, pred_y = load_virus_data(virus, country)

    # Generate graph
    graph_html = generate_trend_graph(actual_x, actual_y, pred_x, pred_y, virus, country)

    return render_template('trends.html', graph_html=graph_html, regions=regions.keys(), viruses=viruses)

def read_csv(file_path):
    """
    Read a CSV file and return its data as a list of lists.
    """
    try:
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            data = list(reader)  # Read entire CSV into a list of lists
        return data
    except Exception as e:
        raise Exception(f"Failed to read CSV file: {str(e)}")


def mergeCombined_GlobalCovid():
    """
    Merge global COVID-19 data with the combined country dataset.
    """
    try:
        # Read datasets
        covid_df = pd.read_csv('data/Global Covid-19 Dataset.csv', encoding='latin1')
        combined_df = pd.read_csv('data/combined_country_data.csv')

        # Merge the dataframes
        merged_df = pd.merge(
            combined_df,
            covid_df[['Country Name', 'Cases', 'Recovered', 'Deaths']],
            left_on='Country',
            right_on='Country Name',
            how='left'
        )

        # Drop the redundant "Country Name" column
        merged_df.drop(columns=['Country Name'], inplace=True)

        # Handle missing values
        merged_df.dropna(subset=['Cases'], inplace=True)
        merged_df['Health Access'].fillna(35, inplace=True)

        # Save the merged DataFrame back to CSV
        merged_df.to_csv('data/combined_country_data.csv', index=False)
        print("combined_country_dataset.csv now includes COVID death, infected, and recovery columns.")
    except Exception as e:
        print(f"Error merging global COVID data: {str(e)}")


def mergeCombined_LiteracyRates():
    """
    Merge literacy rates with the combined country dataset.
    """
    try:
        combined_df = pd.read_csv('data/combined_country_data.csv')
        literacy_df = pd.read_csv('data/Literacy Rate.csv')

        # Drop the existing 'Literacy Rate' column if it exists
        if 'Literacy Rate' in combined_df.columns:
            combined_df.drop('Literacy Rate', axis=1, inplace=True)

        # Merge the dataframes
        merged_df = pd.merge(
            combined_df,
            literacy_df[['Country', 'Literacy Rate']],
            left_on='Country',
            right_on='Country',
            how='left'
        )

        # Handle missing values
        merged_df['Literacy Rate'].fillna(0.85, inplace=True)
        merged_df.drop(merged_df.index[-1], inplace=True)

        # Fill missing values for other columns
        merged_df['GDP per Capita'].fillna(merged_df['GDP per Capita'].mean(), inplace=True)
        merged_df['Population Density'].fillna(merged_df['Population Density'].mean(), inplace=True)
        merged_df['Pollution'].fillna(merged_df['Pollution'].mean(), inplace=True)
        merged_df['Food Insecurity'].fillna(merged_df['Food Insecurity'].mean(), inplace=True)
        merged_df['Travel (Arrival, Tourism)'].fillna(10000, inplace=True)

        # Save the merged DataFrame back to CSV
        merged_df.to_csv('data/combined_country_data.csv', index=False)
    except Exception as e:
        print(f"Error merging literacy rates: {str(e)}")


if __name__ == '__main__':
    # Ensure static directory exists with proper permissions
    static_dir = Path('static')
    static_dir.mkdir(exist_ok=True, mode=0o755)


    # Process the files
    file_configs = [
        ("data/API_EN.ATM.PM25.MC.M3_DS2_en_csv_v2_2052.csv", "2020", "Pollution"),
        ("data/API_EN.POP.DNST_DS2_en_csv_v2_89.csv", "2022", "Population Density"),
        ("data/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_76317.csv", "2023", "GDP per Capita"),
        ("data/API_SE.ADT.LITR.ZS_DS2_en_csv_v2_76004.csv", "2023", "Literacy Rate"),
        ("data/API_SN.ITK.MSFI.ZS_DS2_en_csv_v2_16093.csv", "2022", "Food Insecurity"),
        ("data/API_ST.INT.ARVL_DS2_en_csv_v2_2493.csv", "2020", "Travel (Arrival, Tourism)"),
        ("data/API_SH.UHC.SRVS.CV.XD_DS2_en_csv_v2_15880.csv", "2021", "Health Access"),
    ]

    try:
        # Process and merge data
        combined_df = process_country_data(file_configs)
        combined_df.to_csv("data/combined_country_data.csv", index=False)

        mergeCombined_GlobalCovid()
        mergeCombined_LiteracyRates()

        # Print first few rows to verify
        print("\nFirst few rows of combined data:")
        print(combined_df.head())

        # Print summary of what was processed
        print("\nColumns in combined dataset:", combined_df.columns.tolist())
        print("Total countries processed:", len(combined_df))

        # Train the Random Forest model
        train_model()

        rf = joblib.load("impact_score_model.pkl")
        scaler_X = joblib.load("feature_scaler.pkl")
        scaler_impact = joblib.load("impact_scaler.pkl")

        # Run the Flask app
        app.run(debug=True, threaded=True)
    except Exception as e:
        print(f"Error during initialization: {str(e)}")