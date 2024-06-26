import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from dichotomization.calc_functions import stable_sigmoid
import matplotlib.pyplot as plt


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


class CombinedMaxAUCModel:
    def __init__(self, ind_model, verbose_training=False, K=10,
                 combined_training_iterations=0, refresh_features=False):
        self.ind_model = ind_model
        self.cutoffs = None
        self.individual_weights = None
        self.intercept = None
        self.combined_features = None  # список троек (k, j, xj_cutoff)
        self.combined_weights = None

        self.verbose_training = verbose_training
        self.K = K
        self.combined_training_iterations = combined_training_iterations
        self.refresh_features = refresh_features

    def fit(self, x, y):
        self.cutoffs = self.ind_model.cutoffs
        self.individual_weights = self.ind_model.weights
        self.intercept = self.ind_model.intercept
        self.combined_features = []
        self.combined_weights = []

        for it in range(self.K):
            # добавляем дополнительный комбинированный признак
            # для этого выбираем его из условия максимума AUC
            self.add_combined_feature(x, y, it)
            for it1 in range(self.combined_training_iterations):
                print("Combined iteration", it1 + 1)
                self.make_iteration_combined(x, y)

        if self.refresh_features:
            # повторяем 2 круга: удаляем первые признаки, а потом добавляем заново
            for _ in range(2):
                # удаляем первые 3 комбинированных признаков
                for it in range(3):
                    self.combined_features.pop(0)
                self.combined_weights = self.combined_weights[3:]

                for it1 in range(2 * self.combined_training_iterations):
                    print("Combined iteration", it1 + 1)
                    self.make_iteration_combined(x, y)

                # настраиваем все веса с найденными порогами
                bin_x = self.dichotomize_combined(x)
                logist_reg = LogisticRegression()
                logist_reg.fit(bin_x, y)
                self.combined_weights = logist_reg.coef_.ravel()
                self.intercept = logist_reg.intercept_[0]

                # добавляем 3 признаков заново
                for it in range(3):
                    # добавляем дополнительный комбинированный признак
                    # для этого выбираем его из условия максимума AUC
                    self.add_combined_feature(x, y, it)
                    for it1 in range(self.combined_training_iterations):
                        print("Combined iteration", it1 + 1)
                        self.make_iteration_combined(x, y)

    def make_iteration_combined(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        num_combined_features = len(self.combined_features)
        for s in range(num_combined_features):
            bin_x = self.dichotomize_combined(x)
            # для каждой точки найдем значения решающей функции
            #   (без s-го комбинированного признака)
            weights1 = np.delete(self.combined_weights, s)
            bin_x1 = np.delete(bin_x, s, axis=1)
            # взвешенная сумма для оставшихся признаков
            rest = np.array([np.dot(weights1, bin_x1[i]) for i in range(data_size)])
            # параметры s-го комбинированного признака
            k = self.combined_features[s][0]
            j = self.combined_features[s][1]
            bin_xk = np.where(x[:, k] >= self.cutoffs[k], 1, 0).reshape(-1, 1)
            # обучаем двухфакторную логистическую регрессию:
            #   один признак rest, другой - бинаризованный x_j (с некоторым порогом)
            # для этого перебираем пороги
            grid = np.linspace(np.min(x[:, j]), np.max(x[:, j]), 100, endpoint=False)
            max_auc = None
            optimal_cutoff = None
            for cutoff in grid:
                # бинаризуем данные с выбранным порогом
                bin_xj = np.where(x[:, j] >= cutoff, 1, 0).reshape(-1, 1)
                # обучаем модель логистической регрессии на бинаризованных данных
                log_reg = LogisticRegression(solver='lbfgs', max_iter=10000)
                # два признака: взвешенная сумма остальных признаков и бинарный признак x_k & x_j
                new_x = np.c_[rest, bin_xk * bin_xj]
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
            cur_cutoff = self.combined_features[s][2]
            cur_weight = self.combined_weights[s]

            self.combined_features[s] = (k, j, optimal_cutoff)
            # обновляем веса
            self.combined_weights[s] = wk
            for s1 in range(num_combined_features):
                if s1 != s:
                    self.combined_weights[s1] *= w_rest
            self.intercept = intercept1

            if self.verbose_training:
                print("Предиктор", s + 1, ": AUC = ", max_auc,
                      "; порог =", self.combined_features[s][2], "вместо", cur_cutoff,
                      "; вес =", self.combined_weights[s], "вместо", cur_weight)

    def add_combined_feature(self, x, y, it):
        data_size, num_features = x.shape[0], x.shape[1]
        bin_x = self.dichotomize_combined(x)
        best_auc = 0.0
        best_k = None
        best_j = None
        best_xj_cutoff = None
        for k in range(num_features):
            bin_xk = np.where(x[:, k] >= self.cutoffs[k], 1, 0).reshape(-1, 1)
            # пробуем добавить к k-му признаку какой-нибудь j-й,
            # чтобы максимизировать AUC полученной модели с бинарными признаками
            for j in range(num_features):
                if j == k:
                    continue
                # проверяем, добавлялось ли такое сочетание признаков
                ok = True
                for k1, j1, _ in self.combined_features:
                    if k1 == k and j1 == j:
                        ok = False
                if not ok:
                    continue

                # взвешенная сумма для остальных признаков
                rest = np.array([np.dot(self.combined_weights, bin_x[i]) for i in range(data_size)]).reshape(-1, 1)
                # сетка для перебора порогов
                grid = np.linspace(np.min(x[:, j]), np.max(x[:, j]), 100, endpoint=False)
                max_auc = None
                optimal_cutoff = None
                for cutoff in grid:
                    # бинаризуем данные с выбранным порогом
                    bin_xj = np.where(x[:, j] >= cutoff, 1, 0).reshape(-1, 1)
                    # обучаем модель логистической регрессии на бинаризованных данных
                    log_reg = LogisticRegression(solver='lbfgs', max_iter=10000)
                    # два признака: взвешенная сумма остальных признаков и бинарный признак x_k & x_j
                    new_x = np.c_[rest, bin_xk * bin_xj]
                    log_reg.fit(new_x, y)
                    y_pred_log = log_reg.predict_proba(new_x)[:, 1]
                    # находим AUC для построенной модели
                    fpr, tpr, _ = roc_curve(y, y_pred_log)
                    roc_auc = auc(fpr, tpr)
                    if max_auc is None or roc_auc > max_auc:
                        max_auc = roc_auc
                        optimal_cutoff = cutoff
                # print("k =", k, " j =", j, "xj_cutoff =", optimal_cutoff, "auc =", max_auc)
                if max_auc > best_auc:
                    best_auc = max_auc
                    best_j = j
                    best_k = k
                    best_xj_cutoff = optimal_cutoff

        # добавляем новую комбинацию признаков
        if self.verbose_training:
            print("Новая комбинация №", it + 1, ": k =", best_k, ", j =", best_j,
                  ", порог для xj =", best_xj_cutoff, ", AUC =", best_auc)
        self.combined_features.append((best_k, best_j, best_xj_cutoff))

        # настраиваем все веса с найденными порогами
        bin_x = self.dichotomize_combined(x)
        logist_reg = LogisticRegression()
        logist_reg.fit(bin_x, y)
        self.combined_weights = logist_reg.coef_.ravel()
        self.intercept = logist_reg.intercept_[0]

    def dichotomize_combined(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        bin_x = np.empty_like(x, dtype=int)
        for k in range(num_features):
            bin_x[:, k] = np.where(x[:, k] >= self.cutoffs[k], 1, 0)
        bin_x_combined = np.empty((data_size, len(self.combined_features)), dtype=int)
        for i, (k, j, xj_cutoff) in enumerate(self.combined_features):
            filtering = x[:, j] >= xj_cutoff
            new_feature = bin_x[:, k].astype(bool) & filtering
            bin_x_combined[:, i] = new_feature
        return bin_x_combined

    def predict_proba(self, x):
        bin_x = self.dichotomize_combined(x)
        z = np.dot(bin_x, self.combined_weights) + self.intercept
        probs = np.array([stable_sigmoid(value) for value in z])
        return np.c_[1 - probs, probs]


class SelectedCombinedMaxAUCModel:
    # на вход подается модель AllPairs - то есть признаки с уже найденными порогами
    # модель должна отобрать небольшое число признаков из условия max AUC
    # двухфакторной модели: имеющиеся признаки с найденными весами плюс новый признак
    def __init__(self, model_all_pairs, verbose_training=False, K=10):
        self.model_all_pairs = model_all_pairs
        self.cutoffs = None
        self.individual_weights = None
        self.intercept = None
        self.combined_features = None  # список троек (k, j, xj_cutoff)
        self.combined_weights = None

        self.verbose_training = verbose_training
        self.K = K

    def fit(self, x, y):
        self.cutoffs = self.model_all_pairs.cutoffs
        self.individual_weights = self.model_all_pairs.individual_weights
        self.intercept = self.model_all_pairs.intercept
        self.combined_features = []
        self.combined_weights = []

        for it in range(self.K):
            # добавляем дополнительный комбинированный признак
            # для этого выбираем его из условия максимума AUC
            self.add_combined_feature(x, y, it)

    def add_combined_feature(self, x, y, it):
        data_size, num_features = x.shape[0], x.shape[1]
        bin_x = self.dichotomize_combined(x)
        best_auc = 0.0
        best_k = None
        best_j = None
        best_xj_cutoff = None
        for k, j, xj_cutoff in self.model_all_pairs.combined_features:
            # проверяем, добавлялось ли такое сочетание признаков
            ok = True
            for k1, j1, _ in self.combined_features:
                if k1 == k and j1 == j:
                    ok = False
            if not ok:
                continue

            bin_xk = np.where(x[:, k] >= self.cutoffs[k], 1, 0).reshape(-1, 1)
            bin_xj = np.where(x[:, j] >= xj_cutoff, 1, 0).reshape(-1, 1)
            # взвешенная сумма для остальных признаков
            rest = np.array([np.dot(self.combined_weights, bin_x[i]) for i in range(data_size)]).reshape(-1, 1)
            # обучаем модель логистической регрессии на бинаризованных данных
            log_reg = LogisticRegression(solver='lbfgs', max_iter=10000)
            # два признака: взвешенная сумма остальных признаков и бинарный признак x_k & x_j
            new_x = np.c_[rest, bin_xk * bin_xj]
            log_reg.fit(new_x, y)
            y_pred_log = log_reg.predict_proba(new_x)[:, 1]
            # находим AUC для построенной модели
            fpr, tpr, _ = roc_curve(y, y_pred_log)
            roc_auc = auc(fpr, tpr)
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_j = j
                best_k = k
                best_xj_cutoff = xj_cutoff

        # добавляем новую комбинацию признаков
        if self.verbose_training:
            print("Новая комбинация №", it + 1, ": k =", best_k, ", j =", best_j,
                  ", порог для xj =", best_xj_cutoff, ", AUC =", best_auc)
        self.combined_features.append((best_k, best_j, best_xj_cutoff))

        # настраиваем все веса с найденными порогами
        bin_x = self.dichotomize_combined(x)
        logist_reg = LogisticRegression()
        logist_reg.fit(bin_x, y)
        self.combined_weights = logist_reg.coef_.ravel()
        self.intercept = logist_reg.intercept_[0]

    def dichotomize_combined(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        bin_x = np.empty_like(x, dtype=int)
        for k in range(num_features):
            bin_x[:, k] = np.where(x[:, k] >= self.cutoffs[k], 1, 0)
        bin_x_combined = np.empty((data_size, len(self.combined_features)), dtype=int)
        for i, (k, j, xj_cutoff) in enumerate(self.combined_features):
            filtering = x[:, j] >= xj_cutoff
            new_feature = bin_x[:, k].astype(bool) & filtering
            bin_x_combined[:, i] = new_feature
        return bin_x_combined

    def predict_proba(self, x):
        bin_x = self.dichotomize_combined(x)
        z = np.dot(bin_x, self.combined_weights) + self.intercept
        probs = np.array([stable_sigmoid(value) for value in z])
        return np.c_[1 - probs, probs]


class RandomForest:
    def __init__(self, model_all_pairs, verbose_training=False, K=10):
        self.model_all_pairs = model_all_pairs
        self.cutoffs = None
        self.individual_weights = None
        self.intercept = None
        self.combined_features = None  # список троек (k, j, xj_cutoff)
        self.combined_weights = None
        self.K = K

    def fit(self, x, y):
        self.cutoffs = self.model_all_pairs.cutoffs
        self.individual_weights = self.model_all_pairs.individual_weights
        self.intercept = self.model_all_pairs.intercept
        self.combined_features = self.model_all_pairs.combined_features

        bin_x = self.dichotomize_combined(x)
        #print(bin_x)

        forest = RandomForestClassifier()
        forest.fit(bin_x, y)

        importances = forest.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        """
        feat_labels = [str(k) + "&" + str(j) for k, j, xj_cutoff in self.combined_features]
        for f in range(bin_x.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30,
                                    feat_labels[sorted_indices[f]],
                                    importances[sorted_indices[f]]))
        """
        """
        plt.title('Feature Importance')
        plt.bar(range(bin_x.shape[1]), importances[sorted_indices], align='center')
        plt.xticks(range(bin_x.shape[1]), [feat_labels[i] for i in sorted_indices], rotation=90)
        plt.tight_layout()
        plt.show()
        """

        new_combined_features = []
        for i in sorted_indices:
            if len(new_combined_features) == self.K:
                break
            ok = True
            for k, j, xj_cutoff in new_combined_features:
                if k == self.combined_features[i][1] and j == self.combined_features[i][0]:
                    ok = False
                    break
            if not ok:
                continue
            new_combined_features.append(self.combined_features[i])
        self.combined_features = new_combined_features

        #self.combined_features = [self.combined_features[i] for i in sorted_indices[:self.K]]
        #print(self.combined_features)

        # настраиваем веса
        bin_x = self.dichotomize_combined(x)
        logist_reg = LogisticRegression()
        logist_reg.fit(bin_x, y)
        self.combined_weights = logist_reg.coef_.ravel()
        self.intercept = logist_reg.intercept_[0]

        """
        bin_x = self.dichotomize_combined(x)
        self.forest = RandomForestClassifier()
        self.forest.fit(bin_x, y)
        """


    def dichotomize_combined(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        bin_x = np.empty_like(x, dtype=int)
        for k in range(num_features):
            bin_x[:, k] = np.where(x[:, k] >= self.cutoffs[k], 1, 0)
        bin_x_combined = np.empty((data_size, len(self.combined_features)), dtype=int)
        for i, (k, j, xj_cutoff) in enumerate(self.combined_features):
            filtering = x[:, j] >= xj_cutoff
            new_feature = bin_x[:, k].astype(bool) & filtering
            bin_x_combined[:, i] = new_feature
        return bin_x_combined

    def predict_proba(self, x):
        bin_x = self.dichotomize_combined(x)
        z = np.dot(bin_x, self.combined_weights) + self.intercept
        probs = np.array([stable_sigmoid(value) for value in z])
        return np.c_[1 - probs, probs]
        """
        bin_x = self.dichotomize_combined(x)
        return self.forest.predict_proba(bin_x)
        """
