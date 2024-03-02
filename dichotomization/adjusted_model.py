import numpy as np
from sklearn import metrics as sklearn_metrics
from sklearn.linear_model import LogisticRegression
from initial_model import InitialModel
from adjust_intercept import AdjustIntercept
from tpv_fpv import max_ones_zeros, eps_max_ones_zeros_min_x,\
    eps_max_ones_zeros_max_y, eps_max_ones_zeros_min_y
from calc_functions import stable_sigmoid


class AdjustedModel:
    def __init__(self):
        self.cutoffs = None
        self.weights = None
        self.intercept = None

    def fit(self, x, y, verbose=False):
        data_size, num_features = x.shape[0], x.shape[1]
        initial_model = InitialModel()
        initial_model.fit(x, y)
        self.cutoffs = initial_model.cutoffs
        self.weights = initial_model.weights
        self.intercept = initial_model.intercept

        # Далее перебираем признаки по кругу,
        #   для каждого признака:
        #     исключаем его,
        #     дообучаем модель (только интерсепт) с оставшимися признаками,
        #     выделяем пороговую область
        #       можно визуализировать точки в координатах (x_i, p(x))
        #       пороговая область П состоит из точек
        #         с p(x) чуть меньше порога (0.05),
        #     высота пороговой области подбирается так, чтобы в области
        #       П∩Ф отношение TPV/FPV было максимальным
        #     Ф - фильтрующее свойство, позволяющее предсказать с наибольшей
        #       точностью точки "1" в пороговой области
        #       (например, Ф = {x_i > a_i})
        #     высота пороговой области - это и будет вес i-го признака
        #     выбрав вес, включаем i-й признак с выбранными порогом и весом
        #     подстраиваем интерсепт после обновления весов и порогов

        num_iter = 20
        p_threshold = 0.05
        logit_threshold = stable_sigmoid(p_threshold)
        # TODO: передать порог (сейчас 5%) в качестве входного параметра
        for it in range(num_iter):
            print("Iteration", it + 1)
            self.make_iteration(x, y, logit_threshold, verbose)

    def make_iteration(self, x, y, logit_threshold, verbose, logistic_weights=False, omega=0.1):
        data_size, num_features = x.shape[0], x.shape[1]
        for k in np.random.permutation(num_features):  # range(num_features):
            # производим дихотомизацию
            bin_x = self.dichotomize(x)
            old_xk_cutoff = self.cutoffs[k]
            old_wk = self.weights[k]
            # исключаем k-й признак
            weights1 = np.delete(self.weights, k)
            bin_x1 = np.delete(bin_x, k, axis=1)
            intercept1 = AdjustIntercept(weights1, self.intercept).fit(bin_x1, y)
            # значения решающей функции для каждой точки
            logit = np.array([intercept1 + np.dot(weights1, bin_x1[i])
                              for i in range(data_size)])
            # выделяем пороговую область
            selection = logit < logit_threshold
            logit1 = logit[selection]
            xk = x[selection, k]
            labels = y[selection]
            # находим пороги, обеспечивающие максимум TPV/FPV
            xk_cutoff, min_logit, max_rel = max_ones_zeros(xk, logit1, labels, 10)
            # xk_cutoff, min_logit, max_rel = eps_max_ones_zeros_min_y(xk, logit1, labels, 10, eps=8.0)
            # обновляем порог и вес для k-го признака
            self.cutoffs[k] = omega * xk_cutoff + (1 - omega) * old_xk_cutoff
            self.weights[k] = omega * (logit_threshold - min_logit) + (1 - omega) * old_wk
            # интерсепт пока не трогаем, потому что
            #   для следующего признака он настраивается заново
        # настраиваем интерсепт в конце каждой итерации
        bin_x = self.dichotomize(x)
        self.intercept = AdjustIntercept(self.weights, self.intercept).fit(bin_x, y)
        if verbose:
            print("Пороги:", self.cutoffs)
            print("Веса:", self.weights)
            print("Интерсепт:", self.intercept)

        # настраиваем веса и интерсепт, обучая логистическую регрессию
        if logistic_weights:
            self.fit_logistic(x, y)
            if verbose:
                print("После обучения логистической регрессии:")
                print("Веса:", self.weights)
                print("Интерсепт:", self.intercept)
        # вычисляем AUC
        """
        probs = self.predict_proba(x, y)
        auc = sklearn_metrics.roc_auc_score(y, probs)
        if verbose:
            print("AUC:", auc)
        """

    # взять веса и интерсепт из логистической регрессии при заданных порогах
    def fit_logistic(self, x, y):
        # производим дихотомизацию
        bin_x = self.dichotomize(x)
        logist_reg = LogisticRegression()
        logist_reg.fit(bin_x, y)
        self.weights = logist_reg.coef_.ravel()
        self.intercept = logist_reg.intercept_[0]

    def dichotomize(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        bin_x = x.copy()
        # TODO: исправить, чтобы у bin_x был тип int
        for k in range(num_features):
            bin_x[:, k] = bin_x[:, k] >= self.cutoffs[k]
        return bin_x

    def predict_proba(self, x, y):
        # производим дихотомизацию
        bin_x = self.dichotomize(x)

        z = np.dot(bin_x, self.weights) + self.intercept
        probs = np.array([stable_sigmoid(value) for value in z])
        return probs
