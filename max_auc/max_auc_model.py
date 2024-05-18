import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from dichotomization.calc_functions import stable_sigmoid


class InitialMaxAUCModel:
    def __init__(self):
        self.cutoffs = None
        self.weights = None
        self.intercept = None

    def fit(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        self.cutoffs = np.zeros(num_features)
        # находим пороги в отдельности для каждого k-го признака
        for k in range(num_features):
            # разбиваем диапазон значений k-го признака на 100 частей
            grid = np.linspace(np.min(x[:, k]), np.max(x[:, k]), 100, endpoint=False)
            max_auc = None
            optimal_cutoff = None
            for cutoff in grid:
                # бинаризуем данные с выбранным порогом
                bin_x = np.where(x[:, k] >= cutoff, 1, 0).reshape(-1, 1)
                # обучаем модель логистической регрессии на бинаризованных данных
                model = LogisticRegression(solver='lbfgs', max_iter=10000)
                model.fit(bin_x, y)
                y_pred_log = model.predict_proba(bin_x)[:, 1]
                # находим AUC для построенной модели
                fpr, tpr, _ = roc_curve(y, y_pred_log)
                roc_auc = auc(fpr, tpr)
                if max_auc is None or roc_auc > max_auc:
                    max_auc = roc_auc
                    optimal_cutoff = cutoff
            print("k =", k, " cutoff =", optimal_cutoff)
            self.cutoffs[k] = optimal_cutoff

        # далее обучим модель логистической регрессии на данных, бинаризованных
        #   с найденными порогами, чтобы найти весовые коэффициенты и интерсепт

        # бинаризуем данные
        bin_x = self.dichotomize(x)

        # обучаем логистическую регрессию на бинарных данных
        logist_reg = LogisticRegression()
        logist_reg.fit(bin_x, y)

        self.weights = logist_reg.coef_.ravel()
        self.intercept = logist_reg.intercept_[0]


    def dichotomize(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        bin_x = np.empty_like(x, dtype=int)
        for k in range(num_features):
            bin_x[:, k] = np.where(x[:, k] >= self.cutoffs[k], 1, 0)
        return bin_x


    def predict_proba(self, x):
        bin_x = self.dichotomize(x)
        z = np.dot(bin_x, self.weights) + self.intercept
        probs = np.array([stable_sigmoid(value) for value in z])
        return np.c_[1 - probs, probs]


class IndividualMaxAUCModel:
    def __init__(self, verbose_training=False, training_iterations=20):
        self.cutoffs = None
        self.weights = None
        self.intercept = None

        self.verbose_training = verbose_training
        self.training_iterations = training_iterations


    def fit(self, x, y):
        initial_model = InitialMaxAUCModel()
        initial_model.fit(x, y)
        self.cutoffs = initial_model.cutoffs
        self.weights = initial_model.weights
        self.intercept = initial_model.intercept

        for it in range(self.training_iterations):
            if self.verbose_training:
                print("Iteration", it + 1)
            self.make_iteration(x, y)

        # настраиваем веса с найденными порогами
        bin_x = self.dichotomize(x)
        logist_reg = LogisticRegression()
        logist_reg.fit(bin_x, y)

        self.weights = logist_reg.coef_.ravel()
        self.intercept = logist_reg.intercept_[0]


    def make_iteration(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        for k in range(num_features):
            bin_x = self.dichotomize(x)
            # для каждой точки найдем значения решающей функции
            #   (без k-го признака)
            weights1 = np.delete(self.weights, k)
            bin_x1 = np.delete(bin_x, k, axis=1)
            # взвешенная сумма для оставшихся признаков
            rest = np.array([np.dot(weights1, bin_x1[i]) for i in range(data_size)])
            # обучаем двухфакторную логистическую регрессию:
            #   один признак rest, другой - бинаризованный x_k (с некоторым порогом)
            # для этого перебираем пороги
            grid = np.linspace(np.min(x[:, k]), np.max(x[:, k]), 100, endpoint=False)
            max_auc = None
            optimal_cutoff = None
            w_rest = None
            wk = None
            intercept1 = None
            for cutoff in grid:
                # бинаризуем данные с выбранным порогом
                bin_xk = np.where(x[:, k] >= cutoff, 1, 0).reshape(-1, 1)
                # обучаем модель логистической регрессии на бинаризованных данных
                log_reg = LogisticRegression(solver='lbfgs', max_iter=10000)
                # два признака: взвешенная сумма остальных признаков и бинарный k-й признак
                new_x = np.c_[rest, bin_xk]
                log_reg.fit(new_x, y)
                y_pred_log = log_reg.predict_proba(new_x)[:, 1]
                # находим AUC для построенной модели
                fpr, tpr, _ = roc_curve(y, y_pred_log)
                roc_auc = auc(fpr, tpr)
                if max_auc is None or roc_auc > max_auc:
                    max_auc = roc_auc
                    optimal_cutoff = cutoff
                    w_rest = log_reg.coef_.ravel()[0]
                    wk = log_reg.coef_.ravel()[1]
                    intercept1 = log_reg.intercept_
            #print("k =", k, " cutoff =", optimal_cutoff)
            cur_cutoff = self.cutoffs[k]
            cur_weight = self.weights[k]

            self.cutoffs[k] = optimal_cutoff
            # обновляем веса
            self.weights[k] = wk
            for j in range(num_features):
                if j != k:
                    self.weights[j] *= w_rest
            self.intercept = intercept1

            if self.verbose_training:
                print("Предиктор", k + 1, ": AUC = ", max_auc,
                      "; порог =", self.cutoffs[k], "вместо", cur_cutoff,
                      "; вес =", self.weights[k], "вместо", cur_weight)


    def dichotomize(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        bin_x = np.empty_like(x, dtype=int)
        for k in range(num_features):
            bin_x[:, k] = np.where(x[:, k] >= self.cutoffs[k], 1, 0)
        return bin_x


    def predict_proba(self, x):
        bin_x = self.dichotomize(x)
        z = np.dot(bin_x, self.weights) + self.intercept
        probs = np.array([stable_sigmoid(value) for value in z])
        return np.c_[1 - probs, probs]
