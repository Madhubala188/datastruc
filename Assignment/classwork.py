import numpy as np
x=np.array([[1,2,3],[3,4,5]])
print(x)
print(np.shape(x))
print(np.ndim(x))

y=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(y[1,1,2])
print(np.shape(y))
print(np.ndim(y))
y=np.zeros(5)
print(y)
np.shape(y)
y=np.zeros((4,5))
print(y)
np.shape(y)
y=np.ones((2,3))
print(y)
y=np.full((7,8),11)
x=np.linspace(0,5,10)
print(x)
x2=np.arange(0,5,0.2)
print(x2)
a=1
b=6
amount=50
nopat=np.random.randint(a,b+1,amount)
x=np.random.randn(100)
print(x)
x=np.random.random(10)
print(x)

