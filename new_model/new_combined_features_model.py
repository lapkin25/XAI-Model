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
            #intercept2 = AdjustIntercept(weights1, self.intercept).fit(bin_x1, y,
            #    use_sensitivity=True, p_threshold=self.p0)
            #print(intercept1, intercept2)
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
            # находим порог и вес, обеспечивающие максимум TPV/FPV
            cur_cutoff = self.cutoffs[k] if self.weights[k] > 0 else None
            xk_cutoff, min_logit, max_rel = new_max_ones_zeros(
                xk, logit1, labels, 3, cur_cutoff,
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
            use_sensitivity=False, p_threshold=self.p0)
        #intercept2 = AdjustIntercept(self.weights, self.intercept).fit(bin_x, y,
        #    use_sensitivity=True, p_threshold=self.p0)
        #print(self.intercept, intercept2)

    def fit_intercept(self, x, y):
        bin_x = self.dichotomize(x)
        self.intercept = AdjustIntercept(self.weights, self.intercept).fit(bin_x, y,
            use_sensitivity=True, p_threshold=self.p0)

    def dichotomize(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        bin_x = np.empty_like(x, dtype=int)
        for k in range(num_features):
            bin_x[:, k] = x[:, k] >= self.cutoffs[k]
        return bin_x

    # возвращает для каждой точки две вероятности: "0" и "1"
    def predict_proba(self, x):
        bin_x = self.dichotomize(x)
        z = np.dot(bin_x, self.weights) + self.intercept
        probs = np.array([stable_sigmoid(value) for value in z])
        return np.c_[1 - probs, probs]


class NewCombinedFeaturesModel:
    # p0 - порог отсечения
    # K - число комбинированных признаков
    # delta_a - параметр регуляризации для порога
    # delta_w - параметр регуляризации для веса
    def __init__(self, verbose_training=False, p0=0.05, K=10, delta_a=None, delta_w=None,
                 individual_training_iterations=20, combined_training_iterations=20):
        self.cutoffs = None
        self.individual_weights = None
        self.intercept = None
        self.combined_features = None  # список троек (k, j, xj_cutoff)
        self.combined_weights = None

        self.verbose_training = verbose_training
        self.p0 = p0
        self.K = K
        self.delta_a = delta_a
        self.delta_w = delta_w
        self.individual_training_iterations = individual_training_iterations
        self.combined_training_iterations = combined_training_iterations

    def fit(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]

        ind_model = NewIndividualFeaturesModel(
            verbose_training=self.verbose_training, p0=self.p0,
            delta_a=self.delta_a, delta_w=self.delta_w,
            training_iterations=self.individual_training_iterations)
        ind_model.fit(x, y)
        self.cutoffs = ind_model.cutoffs
        self.individual_weights = ind_model.weights
        self.intercept = -5

        self.combined_features = []
        self.combined_weights = []
        for it in range(self.K):
            # добавляем дополнительный комбинированный признак
            # для этого выбираем его из условия максимума TPV/FPV
            bin_x = self.dichotomize_combined(x)
            # для каждой точки найдем значения решающей функции
            logit = np.array([self.intercept
                + np.dot(self.combined_weights, bin_x[i]) for i in range(data_size)])
            # выделяем пороговую область П
            logit_threshold = inv_sigmoid(self.p0)
            p = np.array([stable_sigmoid(logit[i]) for i in range(data_size)])
            selection = p <= self.p0
            # находим пару признаков, максимизирующую TP/FP
            best_max_rel = None
            best_wj = None
            best_xj_cutoff = None
            best_k = None
            best_j = None
            for k in range(num_features):
                # пробуем добавить к k-му признаку какой-нибудь j-й,
                # чтобы спрогнозировать единицы в области П∩{x_k > a_k}∩Ф
                # с помощью фильтрующего свойства Ф = {x_j > b_j}
                selection_k = x[:, k] >= self.cutoffs[k]
                logit1 = logit[selection & selection_k]
                labels = y[selection & selection_k]
                for j in range(num_features):
                    if j != k:
                        # проверяем, добавлялось ли такое сочетание признаков
                        ok = True
                        for k1, j1, _ in self.combined_features:
                            if k1 == k and j1 == j:
                                ok = False
                        if not ok:
                            continue

                        xj = x[selection & selection_k, j]
                        # находим порог и вес, обеспечивающие максимум TPV/FPV
                        xj_cutoff, min_logit, max_rel = new_max_ones_zeros(
                            xj, logit1, labels, 3, None, None, None, None)
                        if min_logit is None:
                            continue
                        if best_max_rel is None or max_rel > best_max_rel:
                            best_max_rel = max_rel
                            best_wj = logit_threshold - min_logit
                            best_xj_cutoff = xj_cutoff
                            best_k = k
                            best_j = j
            if self.verbose_training:
                print("Новая комбинация №", it + 1, ": k =", best_k, ", j =", best_j,
                      ", порог для xj =", best_xj_cutoff, ", вес для xj =", best_wj,
                      ", TP/FP =", best_max_rel)
            if best_xj_cutoff is None:
                print(len(self.combined_features), "combined features")
                break
            self.combined_features.append((best_k, best_j, best_xj_cutoff))
            self.combined_weights = np.append(self.combined_weights, [best_wj])
            # настраиваем интерсепт в конце каждой итерации
            bin_x = self.dichotomize_combined(x)
            self.intercept = AdjustIntercept(self.combined_weights,
                self.intercept).fit(bin_x, y, use_sensitivity=False,
                p_threshold=self.p0)

            # вспомогательные итерации по настройке комбинированных весов и порогов
            for it1 in range(self.combined_training_iterations):
                if self.verbose_training:
                    print("Combined iteration", it1 + 1)
                self.make_iteration_combined(x, y)

        # повторяем вспомогательные итерации по настройке комбинированных весов и порогов
        for it1 in range(2 * self.combined_training_iterations):
            if self.verbose_training:
                print("Repeat combined iteration", it1 + 1)
            self.make_iteration_combined(x, y)

    def make_iteration_combined(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        num_combined_features = len(self.combined_features)
        for s in range(num_combined_features):
            # производим дихотомизацию
            bin_x = self.dichotomize_combined(x)
            # для каждой точки найдем значения решающей функции
            #   (без s-го комбинированного признака)
            weights1 = np.delete(self.combined_weights, s)
            bin_x1 = np.delete(bin_x, s, axis=1)
            intercept1 = AdjustIntercept(weights1, self.intercept).fit(bin_x1, y,
                use_sensitivity=False, p_threshold=self.p0)
            # значения решающей функции для каждой точки
            logit = np.array([intercept1 + np.dot(weights1, bin_x1[i])
                              for i in range(data_size)])
            # выделяем пороговую область
            p = np.array([stable_sigmoid(logit[i]) for i in range(data_size)])
            logit_threshold = inv_sigmoid(self.p0)
            selection = p <= self.p0
            # параметры s-го комбинированного признака
            k = self.combined_features[s][0]
            j = self.combined_features[s][1]
            selection_k = x[:, k] >= self.cutoffs[k]
            logit1 = logit[selection & selection_k]
            xj = x[selection & selection_k, j]
            labels = y[selection & selection_k]
            # находим порог и вес, обеспечивающие максимум TPV/FPV
            cur_cutoff = self.combined_features[s][2] if self.combined_weights[s] > 0 else None
            xj_cutoff, min_logit, max_rel = new_max_ones_zeros(
                xj, logit1, labels, 3, cur_cutoff,
                logit_threshold - self.combined_weights[s], self.delta_a, self.delta_w)
            if min_logit is not None:
                new_weight = logit_threshold - min_logit
            else:
                new_weight = None
            if self.verbose_training:
                print("Комбинированный предиктор", s + 1, ": TP/FP = ", max_rel,
                      "; порог =", xj_cutoff, "вместо", cur_cutoff,
                      "; вес =", new_weight, "вместо", self.combined_weights[s])
            # обновляем порог и вес для k-го признака
            if xj_cutoff is None:
                self.combined_features[s] = (k, j, 0.0)
                self.combined_weights[s] = 0.0
            else:
                self.combined_features[s] = (k, j, xj_cutoff)
                self.combined_weights[s] = new_weight
        # настраиваем интерсепт в конце каждой итерации
        bin_x = self.dichotomize_combined(x)
        self.intercept = AdjustIntercept(self.combined_weights, self.intercept).fit(bin_x, y,
            use_sensitivity=False, p_threshold=self.p0)

    def fit_intercept(self, x, y):
        bin_x = self.dichotomize_combined(x)
        self.intercept = AdjustIntercept(self.combined_weights, self.intercept).fit(bin_x, y,
            use_sensitivity=True, p_threshold=self.p0)

    def dichotomize_combined(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        bin_x = np.empty_like(x, dtype=int)
        for k in range(num_features):
            bin_x[:, k] = x[:, k] >= self.cutoffs[k]
        bin_x_combined = np.empty((data_size, len(self.combined_features)), dtype=int)
        for i, (k, j, xj_cutoff) in enumerate(self.combined_features):
            filtering = x[:, j] >= xj_cutoff
            new_feature = bin_x[:, k].astype(bool) & filtering
            bin_x_combined[:, i] = new_feature
        return bin_x_combined

    # возвращает для каждой точки две вероятности: "0" и "1"
    def predict_proba(self, x):
        bin_x = self.dichotomize_combined(x)
        z = np.dot(bin_x, self.combined_weights) + self.intercept
        probs = np.array([stable_sigmoid(value) for value in z])
        return np.c_[1 - probs, probs]
