# Flight-Price-Predictor

This project predicts flight prices based on user inputs like airline, source, destination, duration, and days left using a machine learning model trained on real flight data.

## 📁 Project Structure

- `train_model.py`: Data cleaning, preprocessing, training, and model saving.
- `app.py`: Streamlit web app for user input and predictions.
- `random_forest_model.pkl`: Trained model used by the app.
- `Dataset.csv`: Flight pricing dataset (or a link to it if it's too large).

## 📊 Dataset Source

Used a public flight price dataset from [insert source here — e.g., Kaggle](https://www.kaggle.com/). The dataset includes features like:
- Airline
- Source/Destination City
- Stops
- Duration
- Days Left
- Class

## 🛠️ Tech Stack

- Python (pandas, numpy, seaborn, matplotlib)
- Scikit-learn (RandomForestRegressor)
- Streamlit (for web app)
- Joblib (for model saving)
- Git & GitHub

## 📈 Results

Model Performance on test set:
- **RMSE**: ~ 2795
- **R² Score**: 0.9848

### 🔍 Feature Importance (top 15)
![Feature Importance](feature_importance.png)

### 🎯 Residual Plot
![Residual Plot](residuals.png)

## 🚀 Run the App

```bash
streamlit run app.py
