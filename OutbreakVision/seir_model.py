import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
import glob
import os
import pandas as pd

class SEIRModel:
    def __init__(self, S0, E0, I0, R0, beta, sigma, gamma, regions):
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0
        self.beta = beta    # Infection rate
        self.sigma = sigma  # Rate of progression from exposed to infected
        self.gamma = gamma  # Recovery rate
        self.regions = regions  # Dictionary of regions and their populations

    def run_simulation(self, days):
        results = {}
        for region, population in self.regions.items():
            S = [self.S0 * population]
            E = [self.E0 * population]
            I = [self.I0 * population]
            R = [self.R0 * population]

            for _ in range(days):
                new_exposed = self.beta * S[-1] * I[-1] / population
                new_infected = self.sigma * E[-1]
                new_recovered = self.gamma * I[-1]

                S.append(S[-1] - new_exposed)
                E.append(E[-1] + new_exposed - new_infected)
                I.append(I[-1] + new_infected - new_recovered)
                R.append(R[-1] + new_recovered)

            results[region] = I[-1]  # Store the final number of infected individuals

        return results

    def generate_plot(self, days):
        results = self.run_simulation(days)
        
        # Load geographic data from a local file or URL
        world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
        
        # Add infection data to the geographic data
        world['infection_rate'] = world['NAME'].map(results).fillna(0)
        
        # Create static directory if it doesn't exist
        static_dir = Path('static')
        static_dir.mkdir(exist_ok=True)
        
        # Create and save plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        world.boundary.plot(ax=ax)
        world.plot(column='infection_rate', ax=ax, legend=True,
                   legend_kwds={'label': "Infection Rate by Country",
                                'orientation': "horizontal"})
        plt.title('SEIR Model: Infection Spread Across Countries')
        
        # Save plot with absolute path
        img_path = static_dir / 'seir_map.png'
        plt.savefig(img_path)
        plt.close(fig)  # Explicitly close the figure
        
        return 'seir_map.png'

def process_country_data(file_configs):
    """
    Reads multiple CSV files, extracts the specified column for each country,
    and combines them into a single DataFrame.

    :param file_configs: List of tuples (file_path, column_name, output_name)
                         - file_path: Path to the CSV file
                         - column_name: The column containing the data to extract
                         - output_name: The name for the extracted column in the final DataFrame

    :return: A merged DataFrame with 'Country' as the first column and one column per CSV.
    """
    combined_df = None

    for file_path, column_name, output_name in file_configs:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, skiprows=4)


            # Ensure required columns exist
            if column_name not in df.columns:
                print(f"Warning: Column '{column_name}' not found in {file_path}")
                continue

            # Assuming the first column is the country column
            country_column = df.columns[0]

            # Create a temporary DataFrame with Country and the specified column
            temp_df = df[[country_column, column_name]].rename(
                columns={country_column: "Country", column_name: output_name}
            )

            # Merge into the main DataFrame
            if combined_df is None:
                combined_df = temp_df
            else:
                combined_df = combined_df.merge(temp_df, on="Country", how="outer")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    return combined_df if combined_df is not None else pd.DataFrame()