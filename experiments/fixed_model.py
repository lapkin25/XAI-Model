import numpy as np
from scipy.optimize import minimize

# Модель с заданными, фиксированными весовыми коэффициентами
# Настраивается на обучающей выборке только смещение
class FixedModel():
    def __init__(self, weights, initial_intercept):
        self.weights = weights
        self.intercept = initial_intercept

    def fit(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]

        def f_and_df(w0):
            # производные по параметрам модели
            df = 0
            y_pred = np.zeros(data_size)  # выходы, предсказанные моделью
            for i in range(data_size):
                z = w0 + np.dot(self.weights, x[i])  # аргумент сигмоиды
                yp = self.stable_sigmoid(z)  # предсказание выхода
                y_pred[i] = yp
                # factor - множитель при производной решающей функции по параметру
                if y[i] == 0:
                    factor = -1 / (1 - yp)
                elif y[i] == 1:
                    factor = 1 / yp
                else:
                    raise
                factor /= - data_size
                d_sigmoid_features = self.deriv_sigmoid(z)
                df += d_sigmoid_features * factor
            f = self.objective(y, y_pred)
            return f, df
            # return f

        optim_res = minimize(f_and_df, self.intercept, jac=True)
        self.intercept = optim_res.x

    def stable_sigmoid(self, z):
        if z >= 0:
            return 1 / (1 + np.exp(-z))
        else:
            return np.exp(z) / (np.exp(z) + 1)

    def objective(self, y, y_pred):
        y_one_loss = y * np.log(y_pred + 1e-9)
        y_zero_loss = (1 - y) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    # производная d(sigmoid(z))/dz
    def deriv_sigmoid(self, z):
        sig = self.stable_sigmoid(z)
        return sig * (1 - sig)
