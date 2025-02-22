from flask import Flask, render_template
from seir_model import SEIRModel
from pathlib import Path

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
    
    return render_template('index.html', img_path=img_filename)

if __name__ == '__main__':
    # Ensure static directory exists with proper permissions
    static_dir = Path('static')
    static_dir.mkdir(exist_ok=True, mode=0o755)
    
    app.run(debug=True, threaded=True)