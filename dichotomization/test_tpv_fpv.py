import numpy as np
from tpv_fpv import max_ones_zeros
import matplotlib.pyplot as plt

x = np.array([3.1, 2.9, 4.5, 4.6, 4.4, 6.5, 8, 9])
y = np.array([3, 5.1, 10, 5, 7, 5, 2.9, 8])
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
a, b, max_rel = max_ones_zeros(x, y, labels, 2)
print("a =", a, "  b =", b, "  max_rel =", max_rel)

plt.scatter(x, y, c=labels)
plt.show()
