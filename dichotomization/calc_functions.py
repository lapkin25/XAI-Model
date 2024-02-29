import numpy as np

def stable_sigmoid(z):
    if z >= 0:
        return 1 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (np.exp(z) + 1)

# производная d(sigmoid(z))/dz
def deriv_sigmoid(z):
    sig = stable_sigmoid(z)
    return sig * (1 - sig)