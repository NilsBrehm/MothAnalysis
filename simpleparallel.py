import numpy as np
from joblib import Parallel,delayed
from IPython import embed
import time

def f(x,a):
    return x**a

a = np.arange(0,5)

x = np.arange(-500,500,.0001)
y = np.zeros((len(x),len(a)))

t1 = time.time()
for i in range(len(a)):
    y[:,i] = f(x,a[i])


t2 = time.time()
r = Parallel(n_jobs=-1)(delayed(f)(x,a[i]) for i in range(len(a)))
t3 = time.time()

print(t2-t1)
print(t3-t2)

embed()