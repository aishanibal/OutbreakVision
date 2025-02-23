import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib


def train_model():
    # Load the CSV
    file_path = "data/combined_country_data.csv"
    df = pd.read_csv(file_path)

    # Create Impact_Score using the specified calculations
    df["Log_Cases"] = np.log1p(df["Cases"])
    df["Log_Deaths"] = np.log1p(df["Deaths"])

    impact_raw = (
            df["Log_Deaths"] * 5.0 +
            df["Log_Cases"] * 0.7 +
            df["Pollution"] * 1.5 +
            (100 - df["Health Access"]) * 2.0 +
            (100 - df["Literacy Rate"]) * 1.2 +
            (50000 / (df["GDP per Capita"] + 1)) * 1.5
    )

    # Create and save the impact scaler
    scaler_impact = MinMaxScaler(feature_range=(1, 100))
    scaled_impact = scaler_impact.fit_transform(impact_raw.values.reshape(-1, 1)).flatten()
    df["Impact_Score"] = scaled_impact

    # Select feature columns
    feature_cols = [
        "Population Density",
        "GDP per Capita",
        "Literacy Rate",
        "Food Insecurity",
        "Health Access",
        "Travel (Arrival, Tourism)",
        "Pollution"
    ]

    # Extract features (X) and target (y)
    X = df[feature_cols].values
    y = df["Impact_Score"].values

    # Create and save the feature scaler
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Train the model
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_scaled, y)

    # Save scalers and model
    joblib.dump(rf, "impact_score_model.pkl")
    joblib.dump(scaler_X, "feature_scaler.pkl")
    joblib.dump(scaler_impact, "impact_scaler.pkl")

    # Print sample predictions
    sample_indices = np.random.choice(len(X), 5)
    print("\nSample Predictions:")
    for idx in sample_indices:
        pred = rf.predict(X_scaled[idx].reshape(1, -1))[0]
        print(f"Country: {df.iloc[idx]['Country']}")
        print(f"Actual Impact Score: {y[idx]:.2f}")
        print(f"Predicted Impact Score: {pred:.2f}\n")


def predict_impact_score(user_inputs):
    # Load model and scaler
    rf = joblib.load("impact_score_model.pkl")
    scaler_X = joblib.load("feature_scaler.pkl")

    # Scale inputs
    user_inputs_scaled = scaler_X.transform(np.array(user_inputs).reshape(1, -1))

    # Get prediction
    prediction = rf.predict(user_inputs_scaled)[0]

    # Ensure prediction is within bounds
    prediction = np.clip(prediction, 1, 100)

    return prediction