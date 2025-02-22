import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Load the cleaned data
data = pd.read_csv("data/modified_data.csv")

# 2. Print column names for debugging
print("\nAvailable Columns in Dataset:")
print(data.columns.tolist())

# 3. Define the correct feature columns
features = ['GDP per Capita', 'Population Density', 'Food Insecurity', 'Travel (Arrival, Tourism)']
target = 'impact_metric'

# 4. Ensure only available features are selected
features = [col for col in features if col in data.columns]

# 5. Select required columns and drop missing values
data = data[features + [target]].dropna()

# 6. Prepare features (X) and target (y)
X = data[features]
y = data[target]

# 7. Convert non-numeric values if necessary
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# 8. Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Initialize and train the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 10. Make predictions and evaluate the model
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# 11. View feature importances
importances = rf.feature_importances_
print("\nFeature Importances:")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.4f}")

# 12. Visualize predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Impact Metric")
plt.ylabel("Predicted Impact Metric")
plt.title("Actual vs Predicted Impact Metric")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Reference line
plt.show()
