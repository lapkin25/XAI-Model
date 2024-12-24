import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
import math
from dichotomization.calc_functions import stable_sigmoid


class AllPairs:
    def __init__(self, ind_model):
        self.ind_model = ind_model
        self.cutoffs = None
        self.individual_weights = None
        self.intercept = None
        self.combined_features = None  # список троек (k, j, xj_cutoff)
        self.combined_weights = None

    def fit(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        self.cutoffs = self.ind_model.cutoffs
        self.individual_weights = self.ind_model.weights
        self.intercept = self.ind_model.intercept
        self.combined_features = []
        self.combined_weights = []
        probs = []
        for k in range(num_features):
            filtering_k = x[:, k] >= self.cutoffs[k]
            for j in range(num_features):
                if k == j:
                    continue
                # разбиваем диапазон значений j-го признака на 100 частей
                grid = np.linspace(np.min(x[:, j]), np.max(x[:, j]), 100, endpoint=False)
                max_prob = None
                optimal_cutoff = None
                for cutoff in grid:
                    # бинаризуем данные с выбранным порогом
                    filtering_j = x[:, j] >= cutoff
                    # считаем TP, FP
                    cnt = np.sum(filtering_k & filtering_j)
                    tp = np.sum(y[filtering_k & filtering_j])
                    # fp = cnt - tp
                    if tp < 3:
                        continue
                    prob = tp / cnt
                    if max_prob is None or prob > max_prob:
                        max_prob = prob
                        optimal_cutoff = cutoff
                print("k =", k, "j = ", j, "Prob =", max_prob)
                probs.append(max_prob)
                self.combined_features.append((k, j, optimal_cutoff))
                self.combined_weights.append(0.0)
        ind = np.argsort(probs)
        ind = ind[::-1]
        self.combined_features = [self.combined_features[i] for i in ind]

    def fit_auc(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        self.cutoffs = self.ind_model.cutoffs
        self.individual_weights = self.ind_model.weights
        self.intercept = self.ind_model.intercept
        self.combined_features = []
        self.combined_weights = []
        for k in range(num_features):
            filtering_k = x[:, k] >= self.cutoffs[k]
            for j in range(num_features):
                if k == j:
                    continue
                # разбиваем диапазон значений j-го признака на 100 частей
                grid = np.linspace(np.min(x[:, j]), np.max(x[:, j]), 100, endpoint=False)
                max_auc = None
                optimal_cutoff = None
                for cutoff in grid:
                    # бинаризуем данные с выбранным порогом
                    filtering_kj = x[filtering_k, j] >= cutoff
                    y_pred = np.where(filtering_kj, 1, 0)
                    # находим AUC для построенной модели
                    fpr, tpr, _ = roc_curve(y[filtering_k], y_pred)
                    roc_auc = auc(fpr, tpr)
                    if max_auc is None or roc_auc > max_auc:
                        max_auc = roc_auc
                        optimal_cutoff = cutoff
                print("k =", k, "j = ", j, "AUC =", max_auc)
                self.combined_features.append((k, j, optimal_cutoff))
                self.combined_weights.append(0.0)

    # настраивает индивидуальные и комбинированные пороги
    def fit_entropy(self, x, y, simplified=False):
        data_size, num_features = x.shape[0], x.shape[1]
        self.cutoffs = np.zeros(num_features)
        # находим пороги в отдельности для каждого k-го признака
        for k in range(num_features):
            # разбиваем диапазон значений k-го признака на 100 частей
            grid = np.linspace(np.min(x[:, k]), np.max(x[:, k]), 100, endpoint=False)
            min_entropy = None
            optimal_cutoff = None
            for cutoff in grid:
                y_pred = np.where(x[:, k] >= cutoff, 1, 0).reshape(-1, 1)
                cm = confusion_matrix(y, y_pred)
                #print(cm)
                tp = cm[1, 1]
                tn = cm[0, 0]
                fp = cm[0, 1]
                fn = cm[1, 0]
                p1 = (tp + fn) / data_size  # доля реальных единиц в выборке
                p0 = (tn + fp) / data_size  # доля реальных нулей в выборке
                q1 = (tp + fp) / data_size  # доля предсказанных единиц
                q0 = (tn + fn) / data_size  # доля предсказанных нулей
                if p0 == 0 or p1 == 0 or q0 == 0 or q1 == 0:
                    continue

                """
                N0 = cm[0, 0] + cm[1, 0]  # число предсказанных "0"
                N1 = cm[0, 1] + cm[1, 1]  # число предсказанных "1"
                if N1 == 0 or N0 == 0:
                    continue
                p0 = cm[0, 0] / N0
                p1 = cm[1, 1] / N1
                if p0 == 0 or p1 == 0:
                    continue
                """

                """
                # бинаризуем данные с выбранным порогом
                bin_x = np.where(x[:, k] >= cutoff, 1, 0).ravel()
                N1 = np.sum(bin_x)  # число предсказанных "1"
                N0 = data_size - N1  # число предсказанных "0"
                if N1 == 0 or N0 == 0:
                    continue
                p0 = np.sum(y[bin_x == 0] == 0) / N0  # вероятность правильного предсказания "нуля"
                p1 = np.sum(y[bin_x == 1] == 1) / N1  # вероятность правильного предсказания "единицы"
                if p0 == 0 or p1 == 0:
                    continue
                """

                """
                # вероятности классов
                P1 = (cm[1, 0] + cm[1, 1]) / data_size
                P0 = (cm[0, 0] + cm[0, 1]) / data_size
                # априорная энтропия
                E = -P1 * math.log(P1) - P0 * math.log(P0)

                # вероятности классов после срабатывания правила
                P11 = cm[1, 1] / (cm[0, 1] + cm[1, 1])
                P10 = cm[0, 1] / (cm[0, 1] + cm[1, 1])
                # условная энтропия
                if P10 == 0:
                    E1 = -P11 * math.log(P11)
                else:
                    E1 = -P11 * math.log(P11) - P10 * math.log(P10)

                # вероятности классов после не срабатывания правила
                P01 = cm[1, 0] / (cm[1, 0] + cm[0, 0])
                P00 = cm[0, 0] / (cm[1, 0] + cm[0, 0])
                # условная энтропия
                if P01 == 0:
                    E0 = - P00 * math.log(P00)
                else:
                    E0 = -P01 * math.log(P01) - P00 * math.log(P00)

                # information gain
                entropy = -(E - (N1 / data_size) * E1 - (N0 / data_size) * E0)
                """

                #entropy = -(N1/data_size) * math.log(p1) - (N0/data_size) * math.log(p0) + \
                #          (N1/data_size) * math.log(N1/data_size) + (N0/data_size) * math.log(N0/data_size)
                entropy = -p1 * math.log(q1) - p0 * math.log(q0) + \
                    p1 * math.log(p1) + p0 * math.log(p0)
                #print(N0, N1, p0, p1, entropy)
                if min_entropy is None or entropy < min_entropy:
                    min_entropy = entropy
                    optimal_cutoff = cutoff
            print("k =", k, " cutoff =", optimal_cutoff)
            self.cutoffs[k] = optimal_cutoff

        self.individual_weights = np.zeros(num_features)
        self.intercept = 0.0
        # далее находим парные пороги
        self.combined_features = []
        self.combined_weights = []
        for k in range(num_features):
            filtering_k = x[:, k] >= self.cutoffs[k]
            for j in range(num_features):
                if k == j:
                    continue
                # разбиваем диапазон значений j-го признака на 100 частей
                grid = np.linspace(np.min(x[:, j]), np.max(x[:, j]), 100, endpoint=False)
                min_entropy = None
                optimal_cutoff = None
                for cutoff in grid:
                    xj_filtered = x[filtering_k, j]
                    y_filtered = y[filtering_k]
                    y_pred = np.where(xj_filtered >= cutoff, 1, 0).reshape(-1, 1)
                    cm = confusion_matrix(y_filtered, y_pred)
                    N = np.sum(filtering_k)
                    tp = cm[1, 1]
                    tn = cm[0, 0]
                    fp = cm[0, 1]
                    fn = cm[1, 0]
                    p1 = (tp + fn) / N  # доля реальных единиц в выборке
                    p0 = (tn + fp) / N  # доля реальных нулей в выборке
                    q1 = (tp + fp) / N  # доля предсказанных единиц
                    q0 = (tn + fn) / N  # доля предсказанных нулей
                    if p0 == 0 or p1 == 0 or q0 == 0 or q1 == 0:
                        continue

                    """
                    N0 = cm[0, 0] + cm[1, 0]  # число предсказанных "0"
                    N1 = cm[0, 1] + cm[1, 1]  # число предсказанных "1"
                    if N1 == 0 or N0 == 0:
                        continue
                    p0 = cm[0, 0] / N0
                    p1 = cm[1, 1] / N1
                    if p0 == 0 or p1 == 0:
                        continue
                    N = np.sum(filtering_k)
                    """

                    #entropy = -(N1 / N) * math.log(p1) - (N0 / N) * math.log(p0) + \
                    #          (N1 / N) * math.log(N1 / N) + (N0 / N) * math.log(N0 / N)
                    entropy = -p1 * math.log(q1) - p0 * math.log(q0) + \
                              p1 * math.log(p1) + p0 * math.log(p0)
                    if min_entropy is None or entropy < min_entropy:
                        min_entropy = entropy
                        optimal_cutoff = cutoff
                print("k =", k, "j = ", j, " cutoff =", optimal_cutoff)

                # в упрощенной модели берем вторичные пороги такие же, как первичные
                if simplified:
                    optimal_cutoff = self.cutoffs[j]

                self.combined_features.append((k, j, optimal_cutoff))
                self.combined_weights.append(0.0)


# Модель с индивидуальными порогами,
#   найденными из критерия минимума энтропии
class MinEntropyModel:
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
            min_entropy = None
            optimal_cutoff = None
            for cutoff in grid:
                y_pred = np.where(x[:, k] >= cutoff, 1, 0).reshape(-1, 1)
                cm = confusion_matrix(y, y_pred)
                #print(cm)
                N0 = cm[0, 0] + cm[1, 0]  # число предсказанных "0"
                N1 = cm[0, 1] + cm[1, 1]  # число предсказанных "1"
                if N1 == 0 or N0 == 0:
                    continue
                p0 = cm[0, 0] / N0
                p1 = cm[1, 1] / N1
                if p0 == 0 or p1 == 0:
                    continue

                entropy = -(N1/data_size) * math.log(p1) - (N0/data_size) * math.log(p0) + \
                          (N1/data_size) * math.log(N1/data_size) + (N0/data_size) * math.log(N0/data_size)
                #print(N0, N1, p0, p1, entropy)
                if min_entropy is None or entropy < min_entropy:
                    min_entropy = entropy
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
