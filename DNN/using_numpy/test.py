from header import *
import numpy as np

z = np.random.randn(5,2)
(a,b)= z.shape
print(a,b)

print(z)

print(z[0 ,0])
print(z[0][0])
print(z[2][0])
