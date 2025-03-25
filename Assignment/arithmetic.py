import numpy as np
x = np.array([[1, 2],
              [2, 3]])

y = np.array([[1, 1],
              [2, 2]])

print(x)
print(y)
print(np.sin(x))
print(np.cos(x))
print(np.tan(x))
print(np.sqrt(x))
print(np.add(x[0,1],5))
print(np.subtract(x[1,1],2))
print(np.multiply(y[1,1],2))
print(np.divide(y[1,1],2))
n=[1,2,3,4,5,6,7,8,9,10,11,12]
X=np.array(n)
reshape=X.reshape(3,4)
print(reshape)
print(reshape.shape)
a=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
b=a.reshape(4,3)
print(b[1,:])
print(b[:,0])


