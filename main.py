from google.colab import files
uploaded = files.upload()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('d.csv')
print(data.columns)

plt.figure(figsize=(10, 6), dpi=150)
sns.histplot(data['median_house_value'], bins=20, kde=True)
plt.title("Median House Value")
plt.show()

plt.figure(figsize=(10, 6), dpi=150)
sns.histplot(data['median_income'], bins=20, kde=True)
plt.title("Median Income")
plt.show()

plt.figure(figsize=(10, 6), dpi=150)
sns.histplot(data['ocean_proximity'], bins=20, kde=True)
plt.title("Ocean Proximity")
plt.show()

plt.figure(figsize=(10, 6), dpi=150)
sns.scatterplot(x='median_income', y='median_house_value', data=data, hue='ocean_proximity', palette='viridis')
plt.title("Median Income vs Median House Value (Colored by Ocean Proximity)")
plt.show()

plt.figure(figsize=(10, 6), dpi=150)
sns.scatterplot(x='population', y='median_house_value', data=data, hue='ocean_proximity', palette='viridis', s=100)
plt.title("Population vs Median House Value by Ocean Proximity", fontsize=16)
plt.show()

numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()
plt.figure(figsize=(10, 6), dpi=150)
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

encoded_data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)
corr_matrix = encoded_data.corr()
plt.figure(figsize=(10, 6), dpi=150)
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap (Including One-Hot Encoded Ocean Proximity)")
plt.show()

# Predicting User Behavior Class (Logistic Regression)
X = data[['App Usage Time (min/day)', 'Battery Drain (mAh/day)', 'Screen On Time (hours/day)', 'Data Usage (MB/day)']]
y = data['User Behavior Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training logistic regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n")

# Battery Drain Prediction (Linear Regression)
X_battery = data[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Data Usage (MB/day)']]
y_battery = data['Battery Drain (mAh/day)']

# Training linear regression model
battery_model = LinearRegression()
battery_model.fit(X_battery, y_battery)
data['Battery Prediction'] = battery_model.predict(X_battery)

mae = mean_absolute_error(data['Battery Drain (mAh/day)'], data['Battery Prediction'])
mse = mean_squared_error(data['Battery Drain (mAh/day)'], data['Battery Prediction'])
rmse = np.sqrt(mse)
r2 = r2_score(data['Battery Drain (mAh/day)'], data['Battery Prediction'])

print("Regression Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# Plotting Actual vs Predicted Battery Drain
plt.figure(figsize=(10,6), dpi=150)
sns.scatterplot(x='Battery Drain (mAh/day)', y='Battery Prediction', data=data)
plt.title("Actual vs Predicted Battery Drain")
plt.show()
