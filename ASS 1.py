# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import radians, cos, sin, asin, sqrt

#1. Pre-process the dataset.


# Step 2: Load dataset
data = pd.read_csv(r"C:\Users\Asus\Downloads\ML DATSETS\Uber\uber.csv")
print(data.head())

# Step 3: Basic info
print(data.info())
print(data.isnull().sum())

# Drop missing values
data.dropna(inplace=True)

# Remove invalid fare values
data = data[(data['fare_amount'] > 0) & (data['fare_amount'] < 500)]

# Remove invalid coordinates
data = data[(data['pickup_latitude'] >= -90) & (data['pickup_latitude'] <= 90)]
data = data[(data['pickup_longitude'] >= -180) & (data['pickup_longitude'] <= 180)]
data = data[(data['dropoff_latitude'] >= -90) & (data['dropoff_latitude'] <= 90)]
data = data[(data['dropoff_longitude'] >= -180) & (data['dropoff_longitude'] <= 180)]

# Convert pickup_datetime to datetime
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'], errors='coerce')
data.dropna(subset=['pickup_datetime'], inplace=True)

# Extract date-time features
data['year'] = data['pickup_datetime'].dt.year
data['month'] = data['pickup_datetime'].dt.month
data['day'] = data['pickup_datetime'].dt.day
data['hour'] = data['pickup_datetime'].dt.hour









#2. Identify outliers.

# Fare outliers
sns.boxplot(x=data['fare_amount'])
plt.title("Fare Amount Outliers")
plt.show()

# Passenger count outliers
sns.boxplot(x=data['passenger_count'])
plt.title("Passenger Count Outliers")
plt.show()

# Remove unrealistic passenger counts
data = data[(data['passenger_count'] > 0) & (data['passenger_count'] <= 6)]















#3. Check the correlation
# Calculate distance (Haversine formula)
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

data['distance_km'] = data.apply(lambda x: haversine(
    x['pickup_longitude'], x['pickup_latitude'], 
    x['dropoff_longitude'], x['dropoff_latitude']), axis=1)

# Drop irrelevant columns
data = data.drop(['key', 'pickup_datetime'], axis=1, errors='ignore')

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()













#4. Implement linear regression and random forest regression models.

# Define features and target
X = data[['distance_km', 'passenger_count', 'year', 'month', 'day', 'hour']]
y = data['fare_amount']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)









#5. Evaluate the models and compare their respective scores like R2, RMSE, etc
def evaluate_model(y_test, y_pred, model_name):
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{model_name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("-" * 30)

evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest Regression")
