import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import math


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
    def fit_entropy(self, x, y):
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
                entropy = -(N1/data_size) * math.log(p1) - (N0/data_size) * math.log(p0) + \
                          (N1/data_size) * math.log(N1/data_size) + (N0/data_size) * math.log(N0/data_size)
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
                    N0 = cm[0, 0] + cm[1, 0]  # число предсказанных "0"
                    N1 = cm[0, 1] + cm[1, 1]  # число предсказанных "1"
                    if N1 == 0 or N0 == 0:
                        continue
                    p0 = cm[0, 0] / N0
                    p1 = cm[1, 1] / N1
                    if p0 == 0 or p1 == 0:
                        continue
                    N = np.sum(filtering_k)
                    entropy = -(N1 / N) * math.log(p1) - (N0 / N) * math.log(p0) + \
                              (N1 / N) * math.log(N1 / N) + (N0 / N) * math.log(N0 / N)
                    if min_entropy is None or entropy < min_entropy:
                        min_entropy = entropy
                        optimal_cutoff = cutoff
                print("k =", k, "j = ", j, " cutoff =", optimal_cutoff)
                self.combined_features.append((k, j, optimal_cutoff))
                self.combined_weights.append(0.0)
