import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# ======= Load Data =======
file_path = "data/combined_country_data.csv"
df = pd.read_csv(file_path)

# Extract country names
countries = df["Country"].values

# ======= Improved Impact Score Formula =======
df["Log_Cases"] = np.log1p(df["Cases"])  # Log transformation for cases
df["Log_Deaths"] = np.log1p(df["Deaths"])  # Log transformation for deaths

impact_raw = (
    df["Log_Deaths"] * 5.0 +    # Heavily weight deaths
    df["Log_Cases"] * 0.7 +     # Log scale helps large numbers
    df["Pollution"] * 1.5 +     # Increase weight of pollution
    (100 - df["Health Access"]) * 2.0 +  # Low health access increases impact
    (100 - df["Literacy Rate"]) * 1.2 +  # Low literacy rate increases impact
    (50000 / (df["GDP per Capita"] + 1)) * 1.5  # Poorer countries more impacted
)

# Normalize with StandardScaler to preserve variation
scaler_impact = StandardScaler()
impact_scaled = scaler_impact.fit_transform(impact_raw.values.reshape(-1, 1)).flatten()
df["Impact_Score"] = impact_scaled * 10 + 50  # Shift to range roughly [0-100]

# ======= Train Random Forest Model =======
all_numeric_cols = df.select_dtypes(include=[np.number]).columns
feature_cols = all_numeric_cols.difference(["Impact_Score"])

X = df[feature_cols].values
y = df["Impact_Score"].values

X_train, X_test, y_train, y_test, country_train, country_test = train_test_split(
    X, y, countries, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

impact_predictions = rf.predict(X_test)

# ======= Simulating Impact Score Over Time for ALL Countries =======
time_days = np.array([0, 30, 90, 180, 365])

def simulate_impact(initial_impact, decay_factor, time_days):
    return initial_impact * np.exp(-decay_factor * time_days / 365)

decay_factor = 0.2

impact_over_time_results = []
for i, country in enumerate(countries):
    initial_impact = df.loc[df["Country"] == country, "Impact_Score"].values[0]
    impact_over_time = simulate_impact(initial_impact, decay_factor, time_days)
    impact_over_time_results.append([country] + list(impact_over_time))

# Save impact over time results
impact_df = pd.DataFrame(impact_over_time_results, columns=["Country"] + [f"Impact_Day_{t}" for t in time_days])
impact_df.to_csv("impact_overtime.csv", index=False)

print("\nSaved impact over time to 'impact_overtime.csv'.")
