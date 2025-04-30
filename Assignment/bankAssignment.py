# Step 1: Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay


df = pd.read_csv(r"C:\Users\Madhubala\Documents\AI\bank.csv", delimiter=';')


print("Column Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())


df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]



df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'], drop_first=True)


plt.figure(figsize=(12, 8))
sns.heatmap(df3.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()




y = df3['y'].map({'yes': 1, 'no': 0})  # Convert 'yes'/'no' to 1/0
X = df3.drop('y', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)


conf_matrix_log = confusion_matrix(y_test, y_pred_log)
acc_log = accuracy_score(y_test, y_pred_log)

disp = ConfusionMatrixDisplay(conf_matrix_log, display_labels=['No', 'Yes'])
disp.plot(cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

print(f'Logistic Regression Accuracy: {acc_log:.4f}')


knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)

disp = ConfusionMatrixDisplay(conf_matrix_knn, display_labels=['No', 'Yes'])
disp.plot(cmap='Greens')
plt.title('K-Nearest Neighbors Confusion Matrix')
plt.show()

print(f'KNN Accuracy (k=3): {acc_knn:.4f}')


