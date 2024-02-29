import numpy as np
from sklearn.linear_model import LogisticRegression
from initial_model import InitialModel
from adjust_intercept import AdjustIntercept
from calc_functions import stable_sigmoid


class AdjustedModel:
    def __init__(self):
        self.cutoffs = None
        self.weights = None
        self.intercept = None

    def fit(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        initial_model = InitialModel()
        initial_model.fit(x, y)
        self.cutoffs = initial_model.cutoffs
        self.weights = initial_model.weights
        self.intercept = initial_model.intercept

        # Далее перебираем признаки по кругу,
        #   для каждого признака:
        #     исключаем его,
        #     дообучаем модель (только интерсепт) с оставшимися признаками,
        #     выделяем пороговую область
        #       можно визуализировать точки в координатах (x_i, p(x))
        #       пороговая область П состоит из точек
        #         с p(x) чуть меньше порога (0.05),
        #     высота пороговой области подбирается так, чтобы в области
        #       П∩Ф отношение TPV/FPV было максимальным
        #     Ф - фильтрующее свойство, позволяющее предсказать с наибольшей
        #       точностью точки "1" в пороговой области
        #       (например, Ф = {x_i > a_i})
        #     высота пороговой области - это и будет вес i-го признака
        #     выбрав вес, включаем i-й признак с выбранными порогом и весом
        #     подстраиваем интерсепт после обновления весов и порогов

        num_iter = 10
        p_threshold = 0.05
        # TODO: передать порог (сейчас 5%) в качестве входного параметра
        for it in range(num_iter):
            print("Iteration", it + 1)
            for k in range(num_features):
                # производим дихотомизацию
                bin_x = self.dichotomize(x)
                # исключаем k-й признак
                weights1 = np.delete(self.weights, k)
                bin_x1 = np.delete(bin_x, k, axis=1)
                intercept1 = AdjustIntercept(weights1, self.intercept).fit(bin_x1, y)
                # значения решающей функции для каждой точки
                p = np.array([stable_sigmoid(intercept1
                    + np.dot(weights1, bin_x1[i])) for i in range(data_size)])
                # выделяем пороговую область
                selection = p < p_threshold
                p1 = p[selection]
                xk = bin_x[selection, k]

                # интерсепт пока не трогаем, потому что
                #   для следующего признака он настраивается заново

            # не забыть настроить интерсепт в конце каждой итерации







    # взять веса и интерсепт из логистической регрессии при заданных порогах
    def fit_logistic(self, x, y):
        # производим дихотомизацию
        bin_x = self.dichotomize(x)
        logist_reg = LogisticRegression()
        logist_reg.fit(bin_x, y)
        self.weights = logist_reg.coef_.ravel()
        self.intercept = logist_reg.intercept_[0]

    def dichotomize(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        bin_x = x.copy()
        for k in range(num_features):
            bin_x[:, k] = bin_x[:, k] >= self.cutoffs[k]
        return bin_x

    def predict_proba(self, x, y):
        # производим дихотомизацию
        bin_x = self.dichotomize(x)

        z = np.dot(bin_x, self.weights) + self.intercept
        probs = np.array([stable_sigmoid(value) for value in z])
        return probs