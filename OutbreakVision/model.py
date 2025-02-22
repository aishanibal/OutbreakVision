import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset
file_path = "data/combined_country_data.csv"  # Adjust if needed
df = pd.read_csv(file_path)

# Define features and target
feature_columns = ['Pollution', 'Population Density', 'GDP per Capita', 'Food Insecurity',
                   'Travel (Arrival, Tourism)', 'Health Access', 'Literacy Rate']
target_column = 'Cases'

# Extract features and target
X = df[feature_columns].values
y = df[target_column].values.reshape(-1, 1)  # Reshape for PyTorch

# Normalize features using StandardScaler
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Normalize target using MinMaxScaler (to range 0-1)
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define improved neural network with ReLU output
class COVIDPredictor(nn.Module):
    def __init__(self, input_size):
        super(COVIDPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Increased neurons
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)  # Output layer
        self.dropout = nn.Dropout(0.1)  # Prevent overfitting

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # No activation for now
        return torch.relu(x)  # Ensure non-negative output

# Initialize model, loss, and optimizer
input_size = X_train.shape[1]
model = COVIDPredictor(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)
    print(f'Test Loss: {test_loss.item():.6f}')

    # Convert predictions back to original scale
    y_pred_original = scaler_y.inverse_transform(y_pred.numpy())
    y_test_original = scaler_y.inverse_transform(y_test.numpy())

    print(f'Predicted Cases: {y_pred_original.flatten()}')
    print(f'Actual Cases: {y_test_original.flatten()}')
