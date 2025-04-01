import numpy as np
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])
A_inv = np.linalg.inv(A)
A_A_inv = np.dot(A, A_inv)
A_inv_A = np.dot(A_inv, A)
print("Matrix A:")
print(A)
print("\nInverse of A:")
print(A_inv)
print("\nA * A_inv (should be identity matrix):")
print(A_A_inv)
print("\nA_inv * A (should be identity matrix):")
print(A_inv_A)
