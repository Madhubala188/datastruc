import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt(r'C:\Users\Madhubala\PycharmProjects\datastruc\weight-height.csv', delimiter=',', skip_header=1, usecols=(1, 2))
length_inches = data[:, 0]  # Assuming 1st column is height (inches)
weight_pounds = data[:, 1]  # Assuming 2nd column is weight (pounds)
length_cm = length_inches * 2.54
weight_kg = weight_pounds * 0.453592
mean_length = np.mean(length_cm)
mean_weight = np.mean(weight_kg)
print(f"Mean Length: {mean_length:.2f} cm")
print(f"Mean Weight: {mean_weight:.2f} kg")
plt.hist(length_cm, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Length (cm)')
plt.ylabel('Frequency')
plt.title('Distribution of Student Lengths')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()