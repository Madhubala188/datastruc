import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Lasso
X = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in X:
    dice1 = np.random.randint(1, 7, n)
    dice2 = np.random.randint(1, 7, n)
    y = dice1 + dice2  # sum of dice

    h, h2 = np.histogram(y, bins=range(2, 14))  # sums from 2 to 12


    plt.bar(h2[:-1], h / n)
    plt.title(f"Dice Sum Distribution (n = {n})")
    plt.show()
