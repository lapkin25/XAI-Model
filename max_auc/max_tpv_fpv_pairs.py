import numpy as np


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
