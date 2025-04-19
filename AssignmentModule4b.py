import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets  import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
data = pd.read_csv('weight-height.csv')
print(data.head())
plt.scatter(data['Height'], data['Weight'])
plt.show()
X = data[['Height']]
y = data['Weight']
model = LinearRegression()
model.fit(X, y)
plt.scatter(X, y)
plt.plot(X, y, color='red', linewidth=2, label='Prediction')
plt.show()
