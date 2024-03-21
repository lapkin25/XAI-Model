import numpy as np
from dichotomization.initial_model import InitialModel
from dichotomization.adjust_intercept import AdjustIntercept
from dichotomization.calc_functions import stable_sigmoid, inv_sigmoid
from new_tpv_fpv import new_max_ones_zeros


class NewIndividualFeaturesModel:
    # p0 - порог отсечения
    # delta_a - параметр регуляризации для порога
    # delta_w - параметр регуляризации для веса
    def __init__(self, verbose_training=False, p0=0.05, delta_a=None, delta_w=None, training_iterations=20):
        self.cutoffs = None
        self.weights = None
        self.intercept = None

        self.verbose_training = verbose_training
        self.p0 = p0
        self.delta_a = delta_a
        self.delta_w = delta_w
        self.training_iterations = training_iterations

    def fit(self, x, y):
        initial_model = InitialModel()
        initial_model.fit(x, y)
        self.cutoffs = initial_model.cutoffs
        self.weights = initial_model.weights
        self.intercept = initial_model.intercept

        for it in range(self.training_iterations):
            if self.verbose_training:
                print("Iteration", it + 1)
            self.make_iteration(x, y)

    def make_iteration(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        for k in range(num_features):
            # производим дихотомизацию
            bin_x = self.dichotomize(x)
            # для каждой точки найдем значения решающей функции
            #   (без k-го признака)
            weights1 = np.delete(self.weights, k)
            bin_x1 = np.delete(bin_x, k, axis=1)
            intercept1 = AdjustIntercept(weights1, self.intercept).fit(bin_x1, y,
                use_sensitivity=False, p_threshold=self.p0)
                #use_sensitivity=True, p_threshold=self.p0)
            # значения решающей функции для каждой точки
            logit = np.array([intercept1 + np.dot(weights1, bin_x1[i])
                              for i in range(data_size)])
            # выделяем пороговую область
            p = np.array([stable_sigmoid(logit[i]) for i in range(data_size)])
            logit_threshold = inv_sigmoid(self.p0)
            selection = p <= self.p0
            logit1 = logit[selection]
            xk = x[selection, k]
            labels = y[selection]
            # находим пороги, обеспечивающие максимум TPV/FPV
            cur_cutoff = self.cutoffs[k] if self.weights[k] > 0 else None
            xk_cutoff, min_logit, max_rel = new_max_ones_zeros(
                xk, logit1, labels, 7, cur_cutoff,
                logit_threshold - self.weights[k], self.delta_a, self.delta_w)
            if min_logit is not None:
                xk_weight = logit_threshold - min_logit
            else:
                xk_weight = None
            if self.verbose_training:
                print("Предиктор", k + 1, ": TP/FP = ", max_rel,
                      "; порог =", xk_cutoff, "вместо", cur_cutoff,
                      "; вес =", xk_weight, "вместо", self.weights[k])
            # обновляем порог и вес для k-го признака
            if xk_cutoff is None:
                self.cutoffs[k] = 0.0
                self.weights[k] = 0.0
            else:
                self.cutoffs[k] = xk_cutoff
                self.weights[k] = xk_weight
        # настраиваем интерсепт в конце каждой итерации
        bin_x = self.dichotomize(x)
        self.intercept = AdjustIntercept(self.weights, self.intercept).fit(bin_x, y,
            #use_sensitivity=True, p_threshold=self.p0)
            use_sensitivity=False, p_threshold=self.p0)

    def dichotomize(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        bin_x = np.empty_like(x, dtype=int)
        for k in range(num_features):
            bin_x[:, k] = x[:, k] >= self.cutoffs[k]
        return bin_x

    # возвращает для каждой точки две вероятности: "0" и "1"
    # def predict_proba(self, x, y):


class NewCombinedFeaturesModel:
    # p0 - порог отсечения
    # K - число комбинированных признаков
    # delta_a - параметр регуляризации для порога
    # delta_w - параметр регуляризации для веса
    def __init__(self, verbose_training=False, p0=0.05, K=10, delta_a=None, delta_w=None,
                 individual_training_iterations=20):
        self.cutoffs = None
        self.weights = None
        self.intercept = None
        self.combined_features = None  # список троек (k, j, xj_cutoff)
        self.combined_weights = None

        self.verbose_training = verbose_training
        self.p0 = p0
        self.K = K
        self.delta_a = delta_a
        self.delta_w = delta_w
        self.individual_training_iterations = individual_training_iterations

    def fit(self, x, y):
        ind_model = NewIndividualFeaturesModel(
            verbose_training=self.verbose_training, p0=self.p0,
            delta_a=self.delta_a, delta_w=self.delta_w,
            training_iterations=self.individual_training_iterations)
        ind_model.fit(x, y)

    # возвращает для каждой точки две вероятности: "0" и "1"
    # def predict_proba(self, x, y):
