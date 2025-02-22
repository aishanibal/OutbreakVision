from flask import Flask, render_template
from seir_model import SEIRModel
from seir_model import process_country_data
from pathlib import Path
import sqlite3
import csv


# Initialize Flask app with explicit static folder configuration
app = Flask(__name__, static_folder='static', static_url_path='/static')

@app.route('/')
def index():
    # Define regions and their populations
    regions = {
        'China': 1_439_323_776,
        'India': 1_380_004_385,
        'United States': 331_002_651,
        'Indonesia': 273_523_615,
        'Pakistan': 220_892_340,
        # Add more regions as needed
    }
    
    # Initialize model with parameters
    model = SEIRModel(
        S0=0.99,    # Initial susceptible population ratio
        E0=0.01,    # Initial exposed population ratio
        I0=0.0,     # Initial infected population ratio
        R0=0.0,     # Initial recovered population ratio
        beta=0.3,   # Infection rate
        sigma=0.1,  # Rate of progression from exposed to infected
        gamma=0.05, # Recovery rate
        regions=regions
    )
    
    # Generate plot and get relative path
    img_filename = model.generate_plot(days=160)
    csv_data = read_csv("data/combined_country_data.csv")

    
    return render_template('index.html', img_path=img_filename, data=csv_data)


def read_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)  # Read entire CSV into a list of lists
    return data


if __name__ == '__main__':
    # Ensure static directory exists with proper permissions
    file_configs = [
        ("data/API_EN.ATM.PM25.MC.M3_DS2_en_csv_v2_2052.csv", "2020", "Pollution"),
        ("data/API_EN.POP.DNST_DS2_en_csv_v2_89.csv", "2022", "Population Density"),
        ("data/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_76317.csv", "2023", "GDP per Capita"),
        ("data/API_SE.ADT.LITR.ZS_DS2_en_csv_v2_76004.csv", "2023", "Literacy Rate"),
        ("data/API_SN.ITK.MSFI.ZS_DS2_en_csv_v2_16093.csv", "2022", "Food Insecurity"),
        ("data/API_ST.INT.ARVL_DS2_en_csv_v2_2493.csv", "2020", "Travel (Arrival, Tourism)"),
        # Add other files and columns as needed
    ]
    # Process the files
    combined_df = process_country_data(file_configs)

    # Save combined data to a new CSV file
    combined_df.to_csv("data/combined_country_data.csv")

    # Print first few rows to verify
    print("\nFirst few rows of combined data:")
    print(combined_df.head())

    # Print summary of what was processed
    print("\nColumns in combined dataset:", combined_df.columns.tolist())
    print("Total countries processed:", len(combined_df))

    static_dir = Path('static')
    static_dir.mkdir(exist_ok=True, mode=0o755)
    
    app.run(debug=True, threaded=True)