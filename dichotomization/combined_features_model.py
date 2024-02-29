import numpy as np
from adjusted_model import AdjustedModel
from calc_functions import stable_sigmoid
from tpv_fpv import max_ones_zeros


class CombinedFeaturesModel(AdjustedModel):
    def __init__(self):
        super().__init__()
        self.combined_features = None  # список троек (k, j, xj_cutoff)

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
