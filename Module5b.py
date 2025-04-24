import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('50_Startups.csv')
print(df.head())
correlation_matrix = df.corr(numeric_only=True)
print("Correlation matrix:\n", correlation_matrix)


sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
y = df_encoded['Profit']
X = df_encoded.drop('Profit', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)


train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)


print(f"Training RMSE: {train_rmse:.2f}, R²: {train_r2:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}, R²: {test_r2:.2f}")