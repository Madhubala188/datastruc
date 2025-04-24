
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('Auto.csv')
print(df.head())
X = df.drop(columns=['mpg', 'name', 'origin'])
y = df['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


alphas = np.logspace(-4, 4, 100)  # Test a range of alpha values on a logarithmic scale

ridge_r2_scores = []
lasso_r2_scores = []


for alpha in alphas:

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    ridge_preds = ridge.predict(X_test_scaled)
    ridge_r2 = r2_score(y_test, ridge_preds)
    ridge_r2_scores.append(ridge_r2)


    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)
    lasso_preds = lasso.predict(X_test_scaled)
    lasso_r2 = r2_score(y_test, lasso_preds)
    lasso_r2_scores.append(lasso_r2)


plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_r2_scores, label='Ridge Regression', color='black')
plt.plot(alphas, lasso_r2_scores, label='Lasso Regression', color='blue')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R² Score')
plt.title('R² Scores for Ridge and Lasso Regression as Functions of Alpha')
plt.legend()
plt.grid(True)
plt.show()



optimal_ridge_alpha = alphas[np.argmax(ridge_r2_scores)]
optimal_lasso_alpha = alphas[np.argmax(lasso_r2_scores)]

print(f"Optimal Ridge alpha: {optimal_ridge_alpha}")
print(f"Optimal Lasso alpha: {optimal_lasso_alpha}")


ridge_optimal = Ridge(alpha=optimal_ridge_alpha)
ridge_optimal.fit(X_train_scaled, y_train)
ridge_test_preds = ridge_optimal.predict(X_test_scaled)
ridge_test_r2 = r2_score(y_test, ridge_test_preds)
print(f"Ridge Test R²: {ridge_test_r2:.4f}")


lasso_optimal = Lasso(alpha=optimal_lasso_alpha)
lasso_optimal.fit(X_train_scaled, y_train)
lasso_test_preds = lasso_optimal.predict(X_test_scaled)
lasso_test_r2 = r2_score(y_test, lasso_test_preds)
print(f"Lasso Test R²: {lasso_test_r2:.4f}")
