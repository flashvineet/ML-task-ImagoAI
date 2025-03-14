# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

# Step = 1 Load and Inspect Data

# Define file path and load dataset
file_path = "C:\Users\loki\Documents\python\ai_ml_pro\TASK-ML-INTERN.csv" #change according to it
df = pd.read_csv(file_path)

# Display basic info about the dataset
print("\nDataset Info:")
print(df.info())

# Display first few rows
print("\nFirst 5 rows of dataset:")
print(df.head())

# Check for missing values
print("\nMissing Values Count:")
print(df.isnull().sum().sum())  # Should print 0 if no missing values

# Step=2  Data Preprocessing

# Separate features andtarget
X = df.iloc[:, 1:-1]  # Assuming spectral bands are from 2nd column to second-last column
y = df.iloc[:, -1]    # Assuming last column is the target (DON concentration)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step= 3 Data Visualization

# Plot spectral reflectance for first few samples
plt.figure(figsize=(10, 5))
for i in range(5):  
    plt.plot(X.columns, X.iloc[i, :], label=f"Sample {i+1}")
plt.xlabel("Wavelength Bands")
plt.ylabel("Reflectance")
plt.title("Spectral Reflectance of Corn Samples")
plt.legend()
plt.show()

# Heatmap of normalized data
plt.figure(figsize=(10, 5))
sns.heatmap(X_scaled, cmap="coolwarm", xticklabels=False)
plt.title("Heatmap of Spectral Reflectance (Normalized)")
plt.show()

# Step= 4 Dimensionality Reduction (PCA)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components
X_pca = pca.fit_transform(X_scaled)

# Variance explained by PCA components
print("\nExplained Variance by Principal Components:")
print(pca.explained_variance_ratio_)

# Scatter plot of PCA components
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Hyperspectral Data")
plt.colorbar(label="DON Concentration")
plt.show()

# Step = 5 Model Training

# Split data into training(80%) and testing (20%)sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define and train a simple Neural Network (MLP Regressor)
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Stp = 6 Model Evaluation

# Predict on test set
y_pred = model.predict(X_test)

# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# Scatter plot of actual vs predicted values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')  # Diagonal line (ideal predictions)
plt.xlabel("Actual DON Concentration")
plt.ylabel("Predicted DON Concentration")
plt.title("Actual vs. Predicted DON Concentration")
plt.show()

# Step  = 7 Save Model
import joblib
joblib.dump(model, "mlp_regressor_model.pkl")  # Save trained model

print("\n Model training and evaluation completed successfully!")
