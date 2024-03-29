import numpy as np
from sklearn import metrics as sklearn_metrics
from sklearn.linear_model import LogisticRegression
from initial_model import InitialModel
from adjusted_model import AdjustedModel
from adjust_intercept import AdjustIntercept
from calc_functions import stable_sigmoid, inv_sigmoid
from tpv_fpv import max_ones_zeros


class CombinedFeaturesModel(AdjustedModel):
    def __init__(self):
        super().__init__()
        self.combined_features = None  # список троек (k, j, xj_cutoff)
        self.combined_weights = None

    def fit(self, x, y, verbose=True, p_threshold=0.05):
        super().fit(x, y, verbose, p_threshold)
        data_size, num_features = x.shape[0], x.shape[1]
        #p_threshold = 0.05  # TODO: передать как входной параметр
        # logit_threshold = stable_sigmoid(p_threshold)  - грубая ошибка!
        bin_x = self.dichotomize(x)
        logit = np.array([self.intercept +
                          np.dot(self.weights, bin_x[i]) for i in range(data_size)])
        p = np.array([stable_sigmoid(logit[i]) for i in range(data_size)])
        logit_threshold = inv_sigmoid(p_threshold)
        #selection = logit < logit_threshold
        selection = p < p_threshold
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
                    # проверяем, добавлялось ли такое сочетание признаков
                    ok = True
                    for k1, j1, _, _ in combined_features_data:
                        if k1 == k and j1 == j:
                            ok = False
                    if not ok:
                        continue

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
    def fit(self, x, y, verbose=True, logistic_weights=False, p_threshold=0.05, omega=1.0, random_order=True):
        data_size, num_features = x.shape[0], x.shape[1]
        initial_model = InitialModel()
        initial_model.fit(x, y)
        self.cutoffs = initial_model.cutoffs
        self.weights = initial_model.weights
        self.intercept = initial_model.intercept

        num_iter = 20
        #p_threshold = 0.05
        #logit_threshold = stable_sigmoid(p_threshold)
        for it in range(num_iter):
            print("Iteration", it + 1)
            self.make_iteration(x, y, verbose, p_threshold=p_threshold, omega=omega, random_order=random_order)

        self.combined_features = []
        self.combined_weights = []
        num_combined_iter = 15
        num_additional_iter = 5
        for it in range(num_combined_iter):
            # добавляем дополнительный комбинированный признак
            # для этого выбираем его из условия максимума TPV/FPV
            bin_x = self.dichotomize_combined(x)
            logit = np.array([self.intercept +
                              np.dot(np.r_[self.weights, self.combined_weights], bin_x[i])
                              for i in range(data_size)])
            #selection = logit < logit_threshold
            logit_threshold = inv_sigmoid(p_threshold)
            p = np.array([stable_sigmoid(logit[i]) for i in range(data_size)])
            selection = p < p_threshold
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
                        # проверяем, добавлялось ли такое сочетание признаков
                        ok = True
                        for k1, j1, _, in self.combined_features:
                            if k1 == k and j1 == j:
                                ok = False
                        if not ok:
                            continue

                        xj = x[selection & selection_k, j]
                        # находим пороги, обеспечивающие максимум TPV/FPV
                        xj_cutoff, min_logit, max_rel = max_ones_zeros(xj, logit1, labels, 5)
                        if (best_max_rel is None and (min_logit is not None)) or ((min_logit is not None) and max_rel > best_max_rel):
                            best_max_rel = max_rel
                            best_wj = logit_threshold - min_logit
                            best_xj_cutoff = xj_cutoff
                            best_k = k
                            best_j = j
            self.combined_features.append((best_k, best_j, best_xj_cutoff))
            self.combined_weights = np.append(self.combined_weights, [best_wj])
            # настраиваем интерсепт в конце каждой итерации
            bin_x = self.dichotomize_combined(x)
            self.intercept = AdjustIntercept(np.r_[self.weights, self.combined_weights],
                                             self.intercept).fit(bin_x, y)
            # TODO: проделать вспомогательные итерации по настройке весов
            for it1 in range(num_additional_iter):
                print("Additional iteration", it1 + 1)
                self.make_iteration_combined(x, y, verbose, p_threshold=p_threshold, omega=omega, random_order=random_order)
                if logistic_weights:
                    self.fit_logistic_combined(x, y)
        # оптимизируем веса
        if logistic_weights:
            self.fit_logistic_combined(x, y)

    def make_iteration_combined(self, x, y, verbose, omega=0.1, p_threshold=0.05, random_order=True):
        data_size, num_features = x.shape[0], x.shape[1]
        num_combined_features = len(self.combined_features)
        # перенастраиваем веса и пороги индивидуальных признаков
        if random_order:
            range_features = np.random.permutation(num_features)
        else:
            range_features = range(num_features)
        for k in range_features:
            # производим дихотомизацию
            bin_x = self.dichotomize_combined(x)
            old_xk_cutoff = self.cutoffs[k]
            old_wk = self.weights[k]
            # исключаем k-й признак
            weights1 = np.delete(self.weights, k)
            bin_x1 = np.delete(bin_x, k, axis=1)
            intercept1 = AdjustIntercept(np.r_[weights1, self.combined_weights], self.intercept).fit(bin_x1, y)
            # значения решающей функции для каждой точки
            logit = np.array([intercept1 + np.dot(np.r_[weights1, self.combined_weights], bin_x1[i])
                              for i in range(data_size)])
            logit_threshold = inv_sigmoid(p_threshold)
            p = np.array([stable_sigmoid(logit[i]) for i in range(data_size)])
            # выделяем пороговую область
            #selection = logit < logit_threshold
            selection = p < p_threshold
            logit1 = logit[selection]
            xk = x[selection, k]
            labels = y[selection]
            # находим пороги, обеспечивающие максимум TPV/FPV
            xk_cutoff, min_logit, max_rel = max_ones_zeros(xk, logit1, labels, 10)
            #xk_cutoff, min_logit, max_rel = eps_max_ones_zeros_min_x(xk, logit1, labels, 10, eps=6.0)
            # xk_cutoff, min_logit, max_rel = eps_max_ones_zeros_min_y(xk, logit1, labels, 10, eps=8.0)
            # обновляем порог и вес для k-го признака
            self.cutoffs[k] = omega * xk_cutoff + (1 - omega) * old_xk_cutoff
            self.weights[k] = omega * (logit_threshold - min_logit) + (1 - omega) * old_wk
            # интерсепт пока не трогаем, потому что
            #   для следующего признака он настраивается заново
        # перенастраиваем веса и пороги комбинированных признаков
        if random_order:
            range_combined_features = np.random.permutation(num_combined_features)
        else:
            range_combined_features = range(num_combined_features)
        for s in range_combined_features:
            # производим дихотомизацию
            bin_x = self.dichotomize_combined(x)
            # исключаем s-й комбинированный признак
            combined_weights1 = np.delete(self.combined_weights, s)
            bin_x1 = np.delete(bin_x, num_features + s, axis=1)
            intercept1 = AdjustIntercept(np.r_[self.weights, combined_weights1], self.intercept).fit(bin_x1, y)
            # значения решающей функции для каждой точки
            logit = np.array([intercept1 + np.dot(np.r_[self.weights, combined_weights1], bin_x1[i])
                              for i in range(data_size)])
            p = np.array([stable_sigmoid(logit[i]) for i in range(data_size)])
            # параметры s-го комбинированного признака
            k = self.combined_features[s][0]
            j = self.combined_features[s][1]
            # выделяем пороговую область
            #selection = logit < logit_threshold
            selection = p < p_threshold
            selection_k = bin_x[:, k] == 1
            logit1 = logit[selection & selection_k]
            labels = y[selection & selection_k]
            xj = x[selection & selection_k, j]
            # находим пороги, обеспечивающие максимум TPV/FPV
            xj_cutoff, min_logit, max_rel = max_ones_zeros(xj, logit1, labels, 5)
            #xj_cutoff, min_logit, max_rel = eps_max_ones_zeros_min_x(xj, logit1, labels, 5, eps=6.0)
            # обновляем порог и вес для s-го признака
            if xj_cutoff is None:
                self.combined_features[s] = (k, j, 0.0)
                self.combined_weights[s] = 0.0
            else:
                self.combined_features[s] = (k, j, xj_cutoff)
                self.combined_weights[s] = np.max(logit1) - min_logit
            # интерсепт пока не трогаем, потому что
            #   для следующего признака он настраивается заново
        # настраиваем интерсепт в конце каждой итерации
        bin_x = self.dichotomize_combined(x)
        self.intercept = AdjustIntercept(np.r_[self.weights, self.combined_weights], self.intercept).fit(bin_x, y, use_sensitivity=True, p_threshold=p_threshold)
        if verbose:
            print("Пороги:", self.cutoffs)
            print("Веса:", self.weights)
            print("Комбинированные признаки:", self.combined_features)
            print("Веса комбинированных признаков:", self.combined_weights)
            print("Интерсепт:", self.intercept)
        # TODO: перенастроить также и добавленные комбинированные признаки
            # выводим качество модели
            z = np.dot(bin_x, np.r_[self.weights, self.combined_weights]) + self.intercept
            probs = np.array([stable_sigmoid(value) for value in z])
            y_pred = np.where(probs > p_threshold, 1, 0)
            auc = sklearn_metrics.roc_auc_score(y, probs)
            tn, fp, fn, tp = sklearn_metrics.confusion_matrix(y, y_pred).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            print("AUC:", auc, "Sens:", sensitivity, "Spec:", specificity)
            print("tp =", tp, "fn =", fn, "fp =", fp, "tn =", tn)
