import numpy as np
from scipy.optimize import minimize
from calc_functions import stable_sigmoid, inv_sigmoid


# Модель с заданными, фиксированными весовыми коэффициентами
# Настраивается на обучающей выборке только смещение
class AdjustIntercept:
    def __init__(self, weights, initial_intercept):
        self.weights = weights
        self.intercept = initial_intercept

    def fit(self, x, y, p_threshold=0.05, use_sensitivity=False, fix_sensitivity=None):

        use_sensitivity = True
        fix_sensitivity = 0.8

        data_size, num_features = x.shape[0], x.shape[1]

        def f_and_df(w0):
            # производные по параметрам модели
            df = 0.0
            y_pred = np.zeros(data_size)  # выходы, предсказанные моделью
            for i in range(data_size):
                z = w0 + np.dot(self.weights, x[i])  # аргумент сигмоиды
                yp = stable_sigmoid(z)  # предсказание выхода
                y_pred[i] = yp
                df += (yp - y[i]) / data_size
            f = self.objective(y, y_pred)
            return f, df

        if not use_sensitivity:
            optim_res = minimize(f_and_df, self.intercept, jac=True, tol=1e-3)  #, options={'xrtol': 0.1})
            self.intercept = optim_res.x
            return self.intercept[0]
        else:
            # подбираем интерсепт так, чтобы вероятность 0.05 достигалась
            #   при равной чувствительности и специфичности
            logit = [np.dot(self.weights, x[i]) for i in range(data_size)]
            ind = np.argsort(logit)
            tp = np.sum(y)
            fp = data_size - np.sum(y)
            tn = 0
            fn = 0
            logit_threshold = None
            for i in range(data_size):
                # допустим, что все точки с 0-й по i-ю относим к "0",
                #   а все точки после i-й относим к "1"
                if y[ind[i]] == 1:
                    # i-я точка - ложно-отрицательная (до этого была истинно-положительной)
                    fn += 1
                    tp -= 1
                else:
                    # i-я точка - истинно-отрицательная (до этого была ложно-положительной)
                    tn += 1
                    fp -= 1
                sens = tp / (tp + fn)
                spec = tn / (tn + fp)
                if (fix_sensitivity is not None and sens < fix_sensitivity) or spec >= sens:
                    logit_threshold = logit[ind[i]] + 1e-9
                    break
            # решаем уравнение: sigmoid(logit_threshold + intercept) = p_threshold
            return inv_sigmoid(p_threshold) - logit_threshold

    def objective(self, y, y_pred):
        y_one_loss = y * np.log(y_pred + 1e-9)
        y_zero_loss = (1 - y) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)
