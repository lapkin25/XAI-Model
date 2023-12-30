import numpy as np
from scipy.optimize import minimize


# Модель с кусочными весовыми коэффициентами
#   (разные коэффициенты ниже и выше порога)
#   p(x) = σ(w_0 + sum_i [ w^+_i·max(x_i - a_i, 0) + w^-_i·min(x_i - a_i, 0) ])
class PiecewiseModel():
    # weights_plus, weights_minus и intercept - веса и свободный член многофакторной модели
    # a - пороги для каждого признака
    def __init__(self, initial_weights, initial_intercept):
        self.weights_plus = initial_weights
        self.weights_minus = initial_weights
        self.intercept = initial_intercept
        self.a = np.array([0.0 for _ in range(len(initial_weights))])

    # вычисляет предсказание вероятности отнесения к классу "1"
    def calc_p(self, x, w0, w_plus, w_minus, a):
        num_features = x.shape[0]
        z = w0
        for k in range(num_features):
            z += w_plus[k] * max(x[k] - a[k], 0) + w_minus[k] * min(x[k] - a[k], 0)
        return self.stable_sigmoid(z)

    def fit(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]

        def extract_parameters(p):
            # смещение многофакторной модели
            w0 = p[0]
            # веса многофакторной модели
            w_plus = p[1 : num_features + 1]
            w_minus = p[num_features + 1 : 2 * num_features + 1]
            return w0, w_plus, w_minus

        # находит порог из уравнения:
        #   w_+^2 * sum_{x_i > a}[ (a - x_i) * (1 - y_i) ] +
        #   + w_-^2 * sum_{x_i < a}[ (a - x_i) * y_i ] = 0
        def calc_threshold(x, y, w_plus, w_minus):
            n = x.shape[0]
            l = np.min(x)
            r = np.max(x)
            tol = 1e-2
            while (r - l > tol):
                a = (l + r) / 2
                s1 = 0
                s2 = 0
                for i in range(n):
                    if x[i] > a:
                        s1 += (a - x[i]) * (1 - y[i])
                    else:
                        s2 += (a - x[i]) * y[i]
                s = (w_plus ** 2) * s1 + (w_minus ** 2) * s2
                if s < 0:
                    l = a
                else:
                    r = a
            return a

        # вычисляет пороги для каждого признака
        def calc_a(w_plus, w_minus):
            a = np.zeros(num_features)
            for k in range(num_features):
                a[k] = calc_threshold(x[:, k], y, w_plus[k], w_minus[k])
            return a

        def f_and_df(p):
            w0, w_plus, w_minus = extract_parameters(p)
            print('w0 = ', w0)
            print('w+ = ', w_plus)
            print('w- = ', w_minus)
            a = calc_a(w_plus, w_minus)  # вычисляем пороги
            print('a = ', a)
            # производные по параметрам модели
            # не будем их считать, потому что целевая функция зависит еще и от порогов
            #df_w0 = 0
            #df_w_plus = np.zeros(num_features)
            #df_w_minus = np.zeros(num_features)

            y_pred = np.zeros(data_size)
            for i in range(data_size):
                y_pred[i] = self.calc_p(x[i], w0, w_plus, w_minus, a)

            # TODO: вычислить целевую функцию и ее градиент

            f = self.objective(y, y_pred)
            #df = np.concatenate([[df_w0], df_w_plus, df_w_minus])
            #return f, df
            print('f = ', f)
            return f

        # TODO: ...

        w_init = np.concatenate([[self.intercept], self.weights_plus, self.weights_minus])
        #optim_res = minimize(f_and_df, w_init, method='BFGS', jac=True, options={'verbose': 1})
        #optim_res = minimize(f_and_df, w_init, method='BFGS', options={'disp': True})  # или Nelder-Mead?
        optim_res = minimize(f_and_df, w_init, method='Nelder-Mead', options={'disp': True})  # или Nelder-Mead?
        self.intercept, self.weights_plus, self.weights_minus = extract_parameters(optim_res.x)
        self.a = calc_a(self.weights_plus, self.weights_minus)

    def predict(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        y_pred = np.zeros(data_size)
        for i in range(data_size):
            y_pred[i] = self.calc_p(x[i], self.intercept, self.weights_plus, self.weights_minus, self.a)
        return y_pred

    def stable_sigmoid(self, z):
        if z >= 0:
            return 1 / (1 + np.exp(-z))
        else:
            return np.exp(z) / (np.exp(z) + 1)

    def objective(self, y, y_pred):
        y_one_loss = y * np.log(y_pred + 1e-9)
        y_zero_loss = (1 - y) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)
