import numpy as np
from sklearn.linear_model import LogisticRegression
from calc_functions import stable_sigmoid


class InitialModel:
    def __init__(self):
        self.cutoffs = None
        self.weights = None
        self.intercept = None

    def fit(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        self.cutoffs = np.zeros(num_features)
        for k in range(num_features):
            # определяем порог для k-го признака
            grid = np.linspace(np.min(x[:, k]), np.max(x[:, k]), 50, endpoint=False)
            max_r = -1
            optimal_cutoff = None
            for cutoff in grid:
                selection = x[:, k] > cutoff
                # если в области слишком мало точек, пропускаем
                if np.sum(selection) < 10:
                    continue
                dead = y[:] == 1
                survived = y[:] == 0
                tpv = np.sum(selection & dead)
                fpv = np.sum(selection & survived)
                if fpv == 0:
                    continue
                r = tpv / fpv
                if r > max_r:
                    max_r = r
                    optimal_cutoff = cutoff
                self.cutoffs[k] = optimal_cutoff

        # производим дихотомизацию
        bin_x = self.dichotomize(x)

        # обучаем логистическую регрессию на данных с бинарными признаками
        logist_reg = LogisticRegression()
        logist_reg.fit(bin_x, y)
        self.weights = logist_reg.coef_.ravel()
        self.intercept = logist_reg.intercept_[0]

    def dichotomize(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        bin_x = x.copy()
        for k in range(num_features):
            bin_x[:, k] = bin_x[:, k] > self.cutoffs[k]
        return bin_x

    def predict_proba(self, x, y):
        # производим дихотомизацию
        bin_x = self.dichotomize(x)

        z = np.dot(bin_x, self.weights) + self.intercept
        probs = np.array([stable_sigmoid(value) for value in z])
        return probs
