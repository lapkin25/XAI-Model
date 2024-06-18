class ExtractRules:
    def __init__(self, model_all_pairs):
        self.model_all_pairs = model_all_pairs
        self.cutoffs = self.model_all_pairs.cutoffs
        self.combined_features = self.model_all_pairs.combined_features

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
            TP_FP_rel.append(TP / FP)
            print("TP =", TP, "FP =", FP)

        # удаляем наименее качественные правила
        for _ in range(40):
            min_rel = None
            min_i = None
            for i, (k, j, xj_cutoff) in enumerate(self.combined_features):
                if min_rel is None or TP_FP_rel[i] < min_rel:
                    min_rel = TP_FP_rel[i]
                    min_i = i
            del TP_FP_rel[min_i]
            del self.combined_features[i]

        # выводим статистику по правилам
        for i, (k, j, xj_cutoff) in enumerate(self.combined_features):
            print(k, '&', j, "  ", end='')
            print("TP/FP =", TP_FP_rel[i])

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
