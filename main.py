import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt

# Load data
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please download it from Kaggle.")
    print("https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data")
    exit()

# Feature engineering
df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']
features = ['GrLivArea', 'BedroomAbvGr', 'TotalBath']
target = 'SalePrice'
data = df[features + [target]].copy()

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=features + [target])

# Split data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

#print  Output results
print(f"--- Linear Regression Model Results ---")
print(f"Features: {features}")
print(f"Model Coefficients:\n{pd.Series(model.coef_, index=features)}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²: {r2:.4f}")

# Show sample predictions
print("\nSample Predictions vs Actual:")
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head().round(2))

# Visualize predictions grapphs
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted SalePrice")
plt.tight_layout()
plt.show()
