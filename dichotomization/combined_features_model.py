import numpy as np
from sklearn.linear_model import LogisticRegression
from initial_model import InitialModel
from adjusted_model import AdjustedModel
from adjust_intercept import AdjustIntercept
from calc_functions import stable_sigmoid
from tpv_fpv import max_ones_zeros


class CombinedFeaturesModel(AdjustedModel):
    def __init__(self):
        super().__init__()
        self.combined_features = None  # список троек (k, j, xj_cutoff)
        self.combined_weights = None

    def fit(self, x, y, verbose=True):
        super().fit(x, y, verbose)
        data_size, num_features = x.shape[0], x.shape[1]
        p_threshold = 0.05  # TODO: передать как входной параметр
        logit_threshold = stable_sigmoid(p_threshold)
        bin_x = self.dichotomize(x)
        logit = np.array([self.intercept +
                          np.dot(self.weights, bin_x[i]) for i in range(data_size)])
        selection = logit < logit_threshold
        self.combined_features = []
        combined_features_data = []
        # выделяем пороговую область
        for k in range(num_features):
            # пробуем добавить к k-му признаку какой-нибудь j-й,
            # чтобы спрогнозировать единицы в области П∩{x_k > a_k}∩Ф
            # с помощью фильтрующего свойства Ф = {x_j > b_j}
            selection_k = bin_x[:, k] == 1
            logit1 = logit[selection & selection_k]
            labels = y[selection & selection_k]
            for j in range(num_features):
                if j != k:
                    xj = x[selection & selection_k, j]
                    # находим пороги, обеспечивающие максимум TPV/FPV
                    xj_cutoff, min_logit, max_rel = max_ones_zeros(xj, logit1, labels, 5)
                    if verbose:
                        print("k =", k, "j =", j, "b_j =", xj_cutoff, "  w =", logit_threshold - min_logit, "  rel =",  max_rel)
                    # TODO: сначала для упрощения просто взять найденные пороги xj_cutoff
                    #   и добавить комбинированные признаки с этими порогами, а веса найти, обучив логистическую регрессию
                    combined_features_data.append((k, j, xj_cutoff, max_rel))
        combined_features_data.sort(key=lambda d: d[3], reverse=True)
        for i in range(10):
            self.combined_features.append((combined_features_data[i][:-1]))
        # TODO: настроить веса
        # bin_x1 = self.dichotomize_combined(x)
        self.fit_logistic_combined(x, y)

    def dichotomize_combined(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        # num_combined_features = len(self.combined_features)
        bin_x = x.copy()
        for k in range(num_features):
            bin_x[:, k] = bin_x[:, k] >= self.cutoffs[k]
        for k, j, xj_cutoff in self.combined_features:
            filtering = np.array(x[:, j] >= xj_cutoff).astype(int)
            new_feature = bin_x[:, k].astype(int) & filtering
            bin_x = np.c_[bin_x, new_feature]
        return bin_x

    def fit_logistic_combined(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        num_combined_features = len(self.combined_features)
        # производим дихотомизацию
        bin_x = self.dichotomize_combined(x)
        logist_reg = LogisticRegression()
        logist_reg.fit(bin_x, y)
        self.weights = logist_reg.coef_.ravel()[:num_features]
        self.combined_weights = logist_reg.coef_.ravel()[num_features : num_features + num_combined_features]
        self.intercept = logist_reg.intercept_[0]

    def predict_proba(self, x, y):
        # производим дихотомизацию
        bin_x = self.dichotomize_combined(x)

        z = np.dot(bin_x, np.r_[self.weights, self.combined_weights]) + self.intercept
        probs = np.array([stable_sigmoid(value) for value in z])
        return probs


class CombinedFeaturesModel2(CombinedFeaturesModel):
    def fit(self, x, y, verbose=True):
        data_size, num_features = x.shape[0], x.shape[1]
        initial_model = InitialModel()
        initial_model.fit(x, y)
        self.cutoffs = initial_model.cutoffs
        self.weights = initial_model.weights
        self.intercept = initial_model.intercept

        num_iter = 20
        p_threshold = 0.05
        logit_threshold = stable_sigmoid(p_threshold)
        for it in range(num_iter):
            print("Iteration", it + 1)
            self.make_iteration(x, y, logit_threshold, verbose)

        self.combined_features = []
        self.combined_weights = []
        num_combined_iter = 10
        num_additional_iter = 5
        for it in range(num_combined_iter):
            # добавляем дополнительный комбинированный признак
            # для этого выбираем его из условия максимума TPV/FPV
            bin_x = self.dichotomize_combined(x)
            logit = np.array([self.intercept +
                              np.dot(np.r_[self.weights, self.combined_weights], bin_x[i])
                              for i in range(data_size)])
            selection = logit < logit_threshold
            best_max_rel = None
            best_wj = None
            best_xj_cutoff = None
            best_k = None
            best_j = None
            # выделяем пороговую область
            for k in range(num_features):
                # пробуем добавить к k-му признаку какой-нибудь j-й,
                # чтобы спрогнозировать единицы в области П∩{x_k > a_k}∩Ф
                # с помощью фильтрующего свойства Ф = {x_j > b_j}
                selection_k = bin_x[:, k] == 1
                logit1 = logit[selection & selection_k]
                labels = y[selection & selection_k]
                for j in range(num_features):
                    if j != k:
                        xj = x[selection & selection_k, j]
                        # находим пороги, обеспечивающие максимум TPV/FPV
                        xj_cutoff, min_logit, max_rel = max_ones_zeros(xj, logit1, labels, 5)
                        if best_max_rel is None or max_rel is not None and max_rel > best_max_rel:
                            best_max_rel = max_rel
                            best_wj = logit_threshold - min_logit
                            best_xj_cutoff = xj_cutoff
                            best_k = k
                            best_j = j
            self.combined_features.append((best_k, best_j, best_xj_cutoff))
            self.combined_weights.append(best_wj)
            # настраиваем интерсепт в конце каждой итерации
            bin_x = self.dichotomize_combined(x)
            self.intercept = AdjustIntercept(np.r_[self.weights, self.combined_weights],
                                             self.intercept).fit(bin_x, y)
            # TODO: проделать вспомогательные итерации по настройке весов

