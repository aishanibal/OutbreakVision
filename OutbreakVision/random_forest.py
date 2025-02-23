import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt



# 1. Load the CSV (no missing-value handling)
file_path = "data/combined_country_data.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Preview the first few rows
print("DataFrame Head:\n", df.head(), "\n")

# 2. Create a custom Impact Score (range 1–100)
#    Example formula combining 'Deaths', 'Cases', and 'Pollution'
impact_raw = (
    df["Deaths"] * 2.0 +    # Heavily weight deaths
    df["Cases"] * 0.001 +   # Mildly weight total cases
    df["Pollution"] * 0.5   # Mildly weight pollution
)

# Scale raw impact to [1..100]
scaler_impact = MinMaxScaler(feature_range=(1, 100))
impact_scaled = scaler_impact.fit_transform(impact_raw.values.reshape(-1, 1)).flatten()
df["Impact_Score"] = impact_scaled

# 3. Select all numeric columns EXCEPT 'Impact_Score'
all_numeric_cols = df.select_dtypes(include=[np.number]).columns
feature_cols = all_numeric_cols.difference(["Impact_Score"])

print("All Numeric Feature Columns:\n", feature_cols, "\n")

# 4. Extract Features (X) and Target (y)
X = df[feature_cols].values
y = df["Impact_Score"].values

# 5. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. (Optional) Standardize features
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# 7. Train a Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    random_state=42
)
rf.fit(X_train, y_train)

# 8. Evaluate on test set
test_preds = rf.predict(X_test)
mae = mean_absolute_error(y_test, test_preds)
mse = mean_squared_error(y_test, test_preds)
r2 = r2_score(y_test, test_preds)

print("=== Random Forest Performance ===")
print(f"MAE:  {mae:.3f}")
print(f"MSE:  {mse:.3f}")
print(f"R^2:  {r2:.3f}")

# 9. Quick comparison of predictions vs. actual
print("\nSample Predictions vs Actual:")
for i in range(min(5, len(test_preds))):
    print(f"Predicted: {test_preds[i]:.2f}, Actual: {y_test[i]:.2f}")

# 10. Visualize predictions vs. actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, test_preds, alpha=0.7, color="blue")
plt.plot([1, 100], [1, 100], color="red", linestyle="--")  # Ideal diagonal
plt.xlabel("Actual Impact Score")
plt.ylabel("Predicted Impact Score")
plt.title("Random Forest: Predicted vs. Actual Impact Score (All Numeric Columns)")
plt.show()
