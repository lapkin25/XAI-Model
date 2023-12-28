import numpy as np


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

    def fit(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]

        def extract_parameters(p):
            # смещение многофакторной модели
            w0 = p[0]
            # веса многофакторной модели
            w_plus = p[1 : num_features + 1]
            w_minus = p[num_features + 1 : 2 * num_features + 1]
            return w0, w_plus, w_minus

        def f_and_df(p):
            w0, w_plus, w_minus = extract_parameters(p)
            # производные по параметрам модели
            df_w0 = 0
            df_w_plus = np.zeros(num_features)
            df_w_minus = np.zeros(num_features)
            # TODO: вычислить целевую функцию и ее градиент

            return f, df

        # TODO: ...

        optim_res = minimize(f_and_df, w_init, method='BFGS', jac=True, options={'verbose': 1})
        self.intercept, self.weights_plus, self.weights_minus = extract_parameters(optim_res.x)

