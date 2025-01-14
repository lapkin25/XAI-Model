import numpy as np

class ExtractRules:
    def __init__(self, model_all_pairs, K):
        self.model_all_pairs = model_all_pairs
        self.cutoffs = self.model_all_pairs.cutoffs
        self.combined_features = self.model_all_pairs.combined_features
        self.individual_weights = None
        self.combined_weights = None
        self.intercept = None
        self.K = K

    def fit(self, x, y):
        data_size, num_features = x.shape
        # выводим статистику по наблюдениям
        for i in range(data_size):
            if y[i] == 0:
                continue
            print("i =", i, ", y =", y[i], "  ", end='')
            for k, j, xj_cutoff in self.combined_features:
                if x[i, k] >= self.cutoffs[k] and x[i, j] >= xj_cutoff:
                    print(k, '&', j, " ", end='')
            print()
        for i in range(data_size):
            if y[i] == 1:
                continue
            print("i =", i, ", y =", y[i], "  ", end='')
            for k, j, xj_cutoff in self.combined_features:
                if x[i, k] >= self.cutoffs[k] and x[i, j] >= xj_cutoff:
                    print(k, '&', j, " ", end='')
            print()

        # выводим статистику по правилам
        TP_FP_rel = []
        for k, j, xj_cutoff in self.combined_features:
            print(k, '&', j, "  ", end='')
            TP = 0
            FP = 0
            for i in range(data_size):
                # если правило сработало
                if x[i, k] >= self.cutoffs[k] and x[i, j] >= xj_cutoff:
                    if y[i] == 1:
                        TP += 1
                    else:
                        FP += 1
            if FP != 0:
                TP_FP_rel.append(TP / FP)
            else:
                TP_FP_rel.append(1e10)
            print("TP =", TP, "FP =", FP)

        # в каждой паре симметричных правил оставляем наиболее качественное
        deleted = True
        while deleted:
            deleted = False
            for i, (k, j, xj_cutoff) in enumerate(self.combined_features):
                for i1, (k1, j1, xj_cutoff1) in enumerate(self.combined_features):
                    if k == j1 and j == k1 and TP_FP_rel[i] >= TP_FP_rel[i1]:
                        del TP_FP_rel[i1]
                        del self.combined_features[i1]
                        deleted = True
                        break
                if deleted:
                    break

        # удаляем наименее качественные правила
        while len(self.combined_features) > self.K:
            min_rel = None
            min_i = None
            for i, (k, j, xj_cutoff) in enumerate(self.combined_features):
                if min_rel is None or TP_FP_rel[i] < min_rel:
                    min_rel = TP_FP_rel[i]
                    min_i = i
            del TP_FP_rel[min_i]
            del self.combined_features[min_i]

        # выводим статистику по правилам
        for i, (k, j, xj_cutoff) in enumerate(self.combined_features):
            print(k, '&', j, "  ", end='')
            print("TP/FP =", TP_FP_rel[i])

        """
        # повторно выводим статистику по наблюдениям
        # TODO: вынести в отдельную функцию
        for i in range(data_size):
            if y[i] == 0:
                continue
            print("i =", i, ", y =", y[i], "  ", end='')
            for k, j, xj_cutoff in self.combined_features:
                if x[i, k] >= self.cutoffs[k] and x[i, j] >= xj_cutoff:
                    print(k, '&', j, " ", end='')
            print()
        for i in range(data_size):
            if y[i] == 1:
                continue
            print("i =", i, ", y =", y[i], "  ", end='')
            for k, j, xj_cutoff in self.combined_features:
                if x[i, k] >= self.cutoffs[k] and x[i, j] >= xj_cutoff:
                    print(k, '&', j, " ", end='')
            print()
        """

    def predict_proba(self, x):
        data_size, num_features = x.shape
        probs = np.zeros(data_size)
        for i in range(data_size):
            for k, j, xj_cutoff in self.combined_features:
                # если правило сработало
                if x[i, k] >= self.cutoffs[k] and x[i, j] >= xj_cutoff:
                    probs[i] = 1
        return np.c_[1 - probs, probs]

    # сколько правил сработало на каждом из наблюдений
    def count_rules(self, x):
        data_size, num_features = x.shape
        rules_cnt = np.zeros(data_size, dtype=int)
        for i in range(data_size):
            for k, j, xj_cutoff in self.combined_features:
                # если правило сработало
                if x[i, k] >= self.cutoffs[k] and x[i, j] >= xj_cutoff:
                    rules_cnt[i] += 1
        return rules_cnt
