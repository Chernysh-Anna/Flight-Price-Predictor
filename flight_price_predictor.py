import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
#-------
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load data
df = pd.read_csv("Dataset.csv")

# Quick look
print(df.head())
print(df.info())
print(df.describe())


#---Data Cleaning-----------------------------

# Drop nulls 
df.dropna(inplace=True)

# 'departure_time' and 'arrival_time' are categorical (not datetime), so extract directly:
df['departure_time'] = df['departure_time'].astype(str)
df['arrival_time'] = df['arrival_time'].astype(str)

# 'duration' (float) convert to minutes
df['duration_mins'] = df['duration'].apply(lambda x: round(float(x) * 60))

# Map times of day to rough hours (for modeling)
time_mapping = {
    'Early_Morning': 5,
    'Morning': 9,
    'Afternoon': 13,
    'Evening': 17,
    'Night': 21,
    'Late_Night': 1
}
df['departure_hour'] = df['departure_time'].map(time_mapping)
df['arrival_hour'] = df['arrival_time'].map(time_mapping)


#---Data Overview & Visualization-----------------------------

# 1. Check for missing values
print("\n Missing Values:\n")
print(df.isnull().sum())

# 2. Visualize missing data
sns.heatmap(df.isnull(), cbar=False, cmap="Reds")
plt.title("Missing Data Heatmap")
plt.show()

# 3. Show unique airlines
print("\n🛫 Airlines in Dataset:\n")
print(df['airline'].value_counts())
#.Source City
print("\n🛫 Source City:\n")
print(df['source_city'].value_counts())
#.Source City
print("\n🛫 Destination City:\n")
print(df['destination_city'].value_counts())
#.departure_time
print("\n🛫 Departure Time:\n")
print(df['departure_time'].value_counts())
#.departure_time
print("\n🛫 Arrival Time:\n")
print(df['arrival_time'].value_counts())
#.Stops
print("\n🛫 Stops:\n")
print(df['stops'].value_counts())
#.Class
print("\n🛫 Class:\n")
print(df['class'].value_counts())
#.Duration (hours)
print("\n🛫 Duration (hours):\n")
print(df['duration'].value_counts())
#.Days Left
print("\n🛫 Days Left\n")
print(df['days_left'].value_counts())

# 4. Boxplot of Prices by Airline
plt.figure(figsize=(10, 6))
sns.boxplot(x='airline', y='price', data=df)
plt.xticks(rotation=45)
plt.title("Flight Prices by Airline")
plt.ylabel("Price (₹)")
plt.xlabel("Airline")
plt.tight_layout()
plt.show()




#--Trained-----------

# --- One-Hot Encoding ---
df_encoded = pd.get_dummies(df, columns=[
    'airline', 'source_city', 'destination_city', 'stops', 'class'
], drop_first=True)

# --- Feature Selection ---
features = ['departure_hour', 'arrival_hour', 'duration_mins', 'days_left'] + \
           [col for col in df_encoded.columns if col.startswith(('airline_', 'source_city_', 'destination_city_', 'stops_', 'class_'))]

X = df_encoded[features]
y = df_encoded['price']


# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
#-----LinearRegression--------

# --- Train Model ---
lr = LinearRegression()
lr.fit(X_train, y_train)

# --- Predict ---
y_pred = lr.predict(X_test)

"""
#---RandomForestRegressor-----
# Split the data


# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# --- Evaluate ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

#-- Plot Residuals----

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.residplot(x=y_test, y=residuals, lowess=True, line_kws={"color": "red"})
plt.xlabel("Actual Price")
plt.ylabel("Residuals")
plt.title("Residual Plot - Random Forest")
plt.grid(True)
plt.show()


#--  Plot Residuals----

importances = rf.feature_importances_
indices = np.argsort(importances)[-15:]  # Top 15 features

plt.figure(figsize=(10, 6))
plt.title("Top 15 Feature Importances")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Importance")
plt.show()

print(f"\n📊 Model Performance:\nRMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")



# Save  trained Random Forest model
joblib.dump(rf, "random_forest_model.pkl")