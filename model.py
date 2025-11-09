import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Load and prepare dataset
# -------------------------------
df = pd.read_csv("ev.csv")

# Keep relevant columns
df = df[['City', 'Model Year', 'Make', 'Electric Range', 'DOL Vehicle ID']].dropna()

# Aggregate charging demand (approx. number of registered EVs per city/year)
df_grouped = df.groupby(['City', 'Model Year']).size().reset_index(name='EV_Count')

# Feature engineering
df_grouped['Model Year'] = df_grouped['Model Year'].astype(int)
df_grouped['City_Code'] = df_grouped['City'].astype('category').cat.codes

# Features & target
X = df_grouped[['Model Year', 'City_Code']]
y = df_grouped['EV_Count']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Train model
# -------------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# -------------------------------
# Evaluation Metrics
# -------------------------------
def print_metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {label} Metrics:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.3f}")

print_metrics(y_train, y_pred_train, "Training")
print_metrics(y_test, y_pred_test, "Testing")

# -------------------------------
# Feature Importance
# -------------------------------
feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:\n", feat_imp)

# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred_test)
plt.xlabel("Actual EV Demand")
plt.ylabel("Predicted EV Demand")
plt.title("Actual vs Predicted EV Charging Demand")
plt.grid(True)
plt.show()

# Save model output
future_years = pd.DataFrame({
    'Model Year': range(df_grouped['Model Year'].max()+1, df_grouped['Model Year'].max()+6),
    'City_Code': [0]*5  # sample city
})
future_pred = model.predict(future_years)
forecast_df = pd.DataFrame({
    'Model Year': future_years['Model Year'],
    'Predicted_EV_Demand': future_pred.astype(int)
})
forecast_df.to_csv("ev_demand_forecast.csv", index=False)

print("\n🔮 Future Demand Forecast:")
print(forecast_df)
