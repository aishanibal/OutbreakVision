import joblib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


directory = '/Users/katieengel/Library/CloudStorage/OneDrive-Personal/Documents/Programming/OutbreakVision/Google Trends Data/models'
# Load the model
flu_model = joblib.load(os.path.join(directory, 'flu_model.pkl'))

# Load the X and y data
X_data = pd.read_csv(os.path.join(directory, 'X_data.csv'))
X_last_three = pd.read_csv(os.path.join(directory, 'last_x_data.csv'))
full_x_data = pd.concat([X_data, X_last_three], axis=0, ignore_index=True)

pred = flu_model.predict(full_x_data)

  # Predict using the last week data

X_data['DATE'] = pd.to_datetime(X_data['YEAR'].astype(str) + X_data['WEEK'].astype(str) + '1', format='%G%V%u')
full_x_data['DATE'] = pd.to_datetime(full_x_data['YEAR'].astype(str) + full_x_data['WEEK'].astype(str) + '1', format='%G%V%u')

X_data['DATE'] = X_data['DATE'] + pd.Timedelta(weeks=3)
full_x_data['DATE'] = full_x_data['DATE'] + pd.Timedelta(weeks=3)


y_data = pd.read_csv(os.path.join(directory, 'y_data.csv'))

# # Define the directory where you want to save the model
directory = '/Users/katieengel/Library/CloudStorage/OneDrive-Personal/Documents/Programming/OutbreakVision/Google Trends Data/webpage/data'

# Check if the directory exists, if not create it
if not os.path.exists(directory):
    os.makedirs(directory)

# Save the features (X) and target (y) as CSV files
full_x_data['DATE'].to_csv(os.path.join(directory, 'pred_x_flu_united_states.csv'), index=False)
pred = pd.DataFrame(pred, columns=['target_3_weeks_ahead'])
pred.to_csv(os.path.join(directory, 'pred_y_data_flu_united_states.csv'), index=False)

X_data['DATE'].to_csv(os.path.join(directory, 'actual_x_flu_united_states.csv'), index=False)
y_data.to_csv(os.path.join(directory, 'actual_y_data_flu_united_states.csv'), index=False)


# Step 8: Plot Actual vs Predicted Values
plt.figure(figsize=(12, 6))
plt.plot(X_data["DATE"], y_data, label="Actual Cases", color="blue")
plt.plot(full_x_data['DATE'], pred, label="Predicted Flu Cases", color="red")
plt.title("Actual vs Predicted Flu Cases")
plt.xlabel("Year")
plt.ylabel("Number of Flu Cases")
plt.legend()
plt.grid(True)
plt.show()