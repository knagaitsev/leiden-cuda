import numpy as np
import mycuda

a = np.ones(10, dtype=np.float32)
b = np.ones(10, dtype=np.float32) * 2
c = mycuda.add(a, b)
print(c)
