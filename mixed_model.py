import numpy as np
from scipy.optimize import minimize

# Смешанная модель, состоящая из многофакторной логистической регрессии
#   и нескольких однофакторных логистических регрессий, взятых с весами:
#   p(x) = ω_1·σ(a_1·x_{s_1} + b_1) + ... + ω_m·σ(a_m·x_{s_m} + b_m) +
#     + (1 - ω)·σ(w_0 + w_1·x_1 + ... + w_p·x_p)
# Оптимизация с ограничением: ω_1 + ... + ω_m = ω
# Образец кода: https://www.dmitrymakarov.ru/opt/logistic-regression-05/
class MixedModel():

    # selected_features - выделенные признаки, по которым строится
    #   однофакторная логистическая модель
    # weights и intercept - веса и свободный член многофакторной модели
    # a, b - веса и свободные члены однофакторных моделей
    # omega - суммарная доля однофакторных моделей в общей модели
    def __init__(self, selected_features, omega, initial_weights, initial_intercept):
        self.selected_features = selected_features
        self.num_selected_features = len(selected_features)
        self.omega = omega
        self.weights = initial_weights
        self.intercept = initial_intercept
        self.a = np.array([1 for _ in range(self.num_selected_features)])
        self.b = np.array([0 for _ in range(self.num_selected_features)])
        self.omega_coefficients = np.concatenate([[omega], np.array([0 for _ in range(self.num_selected_features - 1)])])
        # self.loss_history = []

    # метод .fit() необходим для обучения модели
    def fit(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        num_selected_features = self.num_selected_features
        omega = self.omega
        # loss_history = []

        def extract_parameters(p):
            # смещение многофакторной модели
            w0 = p[0]
            # веса многофакторной модели
            w = p[1 : num_features + 1]
            # коэффициенты однофакторных моделей
            a = p[num_features + 1 : num_features + 1 + num_selected_features]
            # смещения однофакторных моделей
            b = p[num_features + 1 + num_selected_features : num_features + 1 + 2 * num_selected_features]
            # веса при однофакторных моделях
            omega_selected = p[num_features + 1 + 2 * num_selected_features : num_features + 3 * num_selected_features]
            # коэффициент omega_k для последнего выделенного признака вычисляем
            omega_selected = np.append(omega_selected, omega - np.sum(omega_selected))
            return w0, w, a, b, omega_selected

        def f_and_df(p):
            # nonlocal num_features
            # nonlocal num_selected_features
            # nonlocal omega
            w0, w, a, b, omega_selected = extract_parameters(p)
            # производные по параметрам модели
            df_intercept = 0
            df_weights = np.zeros(num_features)
            df_a = np.zeros(num_selected_features)
            df_b = np.zeros(num_selected_features)
            df_omega_coefficients = np.zeros(num_selected_features - 1)
            zk = np.zeros(num_selected_features)  # аргументы сигмоид для выделенных признаков
            y_pred = np.zeros(data_size)  # выходы, предсказанные моделью
            for i in range(data_size):
                yp = 0  # предсказание выхода
                for k in range(num_selected_features):
                    j = self.selected_features[k]  # номер выделенного признака
                    zk[k] = a[k] * x[i][j] + b[k]  # аргумент сигмоиды
                    yp += omega_selected[k] * self.stable_sigmoid(zk[k])
                z = w0 + np.dot(w, x[i])  # аргумент сигмоиды
                yp += (1 - omega) * self.stable_sigmoid(z)
                y_pred[i] = yp
                # factor - множитель при производной решающей функции по параметру
                if y[i] == 0:
                    factor = -1 / (1 - yp)
                elif y[i] == 1:
                    factor = 1 / yp
                else:
                    raise
                factor /= data_size
                d_sigmoid_features = self.deriv_sigmoid(z)
                df_intercept += (1 - omega) * d_sigmoid_features * factor
                for k in range(num_features):
                    df_weights[k] += (1 - omega) * d_sigmoid_features * x[i][k] * factor
                for k in range(num_selected_features):
                    d_sigmoid_selected = self.deriv_sigmoid(zk[k])
                    j = self.selected_features[k]  # номер выделенного признака
                    df_a[k] += omega_selected[k] * d_sigmoid_selected * x[i][j] * factor
                    df_b[k] += omega_selected[k] * d_sigmoid_selected * factor
                for k in range(num_selected_features - 1):
                    df_omega_coefficients[k] += self.stable_sigmoid(zk[k]) * factor

            f = self.objective(y, y_pred)
            #df = np.zeros(1 + num_features + 3 * num_selected_features - 1)
            df = np.concatenate([[df_intercept], df_weights, df_a, df_b, df_omega_coefficients])
            #return f, df
            return f

        w_init = np.concatenate([[self.intercept], self.weights, self.a, self.b, self.omega_coefficients])
        # optim_res = minimize(f_and_df, w_init, method='BFGS', jac=True, options={'disp': True})
        optim_res = minimize(f_and_df, w_init, method='BFGS', options={'disp': True})
        self.intercept, self.weights, self.a, self.b, self.omega_coefficients = extract_parameters(optim_res.x)


    # метод .predict() делает прогноз вероятности отнесения к классу "1" с помощью обученной модели
    def predict(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        num_selected_features = self.num_selected_features
        omega = self.omega
        w0, w, a, b, omega_selected = self.intercept, self.weights, self.a, self.b, self.omega_coefficients[:-1]
        # TODO: можно вынести повторяющийся код
        zk = np.zeros(num_selected_features)  # аргументы сигмоид для выделенных признаков
        y_pred = np.zeros(data_size)  # выходы, предсказанные моделью
        for i in data_size:
            yp = 0  # предсказание выхода
            for k in range(num_selected_features):
                j = self.selected_features[k]  # номер выделенного признака
                zk[k] = a[k] * x[i][j] + b[k]  # аргумент сигмоиды
                yp += omega_selected[k] * self.stable_sigmoid(zk[k])
            z = w0 + np.dot(w, x[i])  # аргумент сигмоиды
            yp += (1 - omega) * self.stable_sigmoid(z)
            y_pred[i] = yp
        return y_pred

    def objective(self, y, y_pred):
        try:
            y_one_loss = y * np.log(y_pred + 1e-9)
        finally:
            print(np.min(y_pred), np.max(y_pred))
        y_zero_loss = (1 - y) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def stable_sigmoid(self, z):
        if z >= 0:
            return 1 / (1 + np.exp(-z))
        else:
            return np.exp(z) / (np.exp(z) + 1)

    # производная d(sigmoid(z))/dz
    def deriv_sigmoid(self, z):
        sig = self.stable_sigmoid(z)
        return sig * (1 - sig)


# TODO: добавить расчет AUC, порога отсечения и прочих метрик
