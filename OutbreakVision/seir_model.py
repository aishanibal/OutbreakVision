import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path

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
        
        # Load geographic data
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        
        # Add infection data to the geographic data
        world['infection_rate'] = world['name'].map(results).fillna(0)
        
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