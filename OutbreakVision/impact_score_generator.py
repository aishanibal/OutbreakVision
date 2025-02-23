def generate_impact_scores(decay_factor):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # Load Data
    file_path = "data/combined_country_data.csv"
    df = pd.read_csv(file_path)

    # Compute Impact Score
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

    scaler_impact = StandardScaler()
    impact_scaled = scaler_impact.fit_transform(impact_raw.values.reshape(-1, 1)).flatten()
    df["Impact_Score"] = impact_scaled * 10 + 50

    # Simulate Impact Score Over Time Using Decay Factor
    time_days = np.array([0, 30, 90, 180, 365])

    def simulate_impact(initial_impact, decay_factor, time_days):
        return initial_impact * np.exp(-decay_factor * time_days / 365)

    impact_over_time_results = []
    for i, country in enumerate(df["Country"]):
        initial_impact = df.loc[df["Country"] == country, "Impact_Score"].values[0]
        impact_over_time = simulate_impact(initial_impact, decay_factor, time_days)
        impact_over_time_results.append([country] + list(impact_over_time))

    impact_df = pd.DataFrame(impact_over_time_results, columns=["Country"] + [f"Impact_Day_{t}" for t in time_days])
    impact_df.to_csv("impact_overtime.csv", index=False)

    print("\nUpdated impact scores in 'impact_overtime.csv' using decay factor:", decay_factor)
