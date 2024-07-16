import numpy as np
from sklearn.linear_model import LogisticRegression
from dichotomization.calc_functions import stable_sigmoid
from max_auc.max_auc_model import InitialMaxAUCModel
from max_auc.max_tpv_fpv_pairs import MinEntropyModel


# Модель - логистическая регрессия, зависящая от преобразованных признаков z_i
# Каждый преобразованный признак - это конъюнкция дихотомизированного i-го признака x'_i
#   и выхода вспомогательной логистической регрессии, построенной на остальных признаках,
#   обученной на части выборки, состоящей из точек, для которых x'_i = 1
class CombinedModel:
    def __init__(self, ind_model, threshold=0.04, method="auc"):
        # модель с порогами для отдельных признаков
        self.ind_model = ind_model
        # параметры логистической регрессии, зависящей от z_i
        self.weights = None
        self.intercept = None
        # пороги дихотомизации для каждого признака
        self.cutoffs = None
        # порог для предсказанной вероятности
        self.threshold = threshold
        # пороги дихотомизации для вспомогательных логистических регрессий
        self.middle_cutoffs = None
        # параметры всопогательных логистических регрессий
        self.middle_weights = None
        self.middle_intercept = None
        # метод нахождения порогов
        self.method = method

    def fit(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]

        self.middle_cutoffs = np.zeros((num_features, num_features))
        self.middle_weights = np.zeros((num_features, num_features))
        self.middle_intercept = np.zeros(num_features)

        # пороги такие же, как в индивидуальной модели
        self.cutoffs = self.ind_model.cutoffs

        # Сначала обучаем вспомогательные модели
        for k in range(num_features):
            print("Настройка вспомогательной модели для k =", k)
            # формируем данные, на которых будем обучать модель
            filtering_k = x[:, k] >= self.cutoffs[k]
            mid_x = np.c_[x[filtering_k, :k], x[filtering_k, k+1:]]
            mid_y = y[filtering_k]
            # обучаем модель с порогами
            if self.method == "auc":
                model = InitialMaxAUCModel()
            elif self.method == "entropy":
                model = MinEntropyModel()
            model.fit(mid_x, mid_y)
            # сохраняем найденные параметры модели
            self.middle_cutoffs[k, :] = np.concatenate((model.cutoffs[:k], [0.0], model.cutoffs[k:]))
            self.middle_weights[k, :] = np.concatenate((model.weights[:k], [0.0], model.weights[k:]))
            self.middle_intercept[k] = model.intercept

        z = self.transformed_features(x)

        # Теперь обучаем комбинированную модель на построенных преобразованных признаках
        logist_reg = LogisticRegression()
        logist_reg.fit(z, y)
        self.weights = logist_reg.coef_.ravel()
        self.intercept = logist_reg.intercept_[0]

    def predict_proba(self, x):
        z = self.transformed_features(x)
        # z_i - дихотомизированный преобразованный признак
        logits = np.dot(z, self.weights) + self.intercept
        probs = np.array([stable_sigmoid(value) for value in logits])
        return np.c_[1 - probs, probs]

    # Используя параметры модели, посчитать преобразованные переменные
    def transformed_features(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        z = np.zeros((data_size, num_features), dtype=int)

        for i in range(data_size):
            for k in range(num_features):
                # если x[i, k] не за порогом, оставляем z[i, k] равным 0
                # иначе считаем выход вспомогательной логистической регрессии
                if x[i, k] >= self.cutoffs[k]:
                    # вычисляем дихотомизированные признаки внутри вспомогательной модели
                    bin_x = np.zeros(num_features)
                    # bin_x[k] останется равным 0 - этого признака во вспомогательной модели нет
                    for j in range(num_features):
                        if j == k:
                            continue
                        if x[i, j] >= self.middle_cutoffs[k, j]:
                            bin_x[j] = 1
                    # находим выход вспомогательной модели
                    logit = np.dot(bin_x, self.middle_weights[k, :]) + self.middle_intercept[k]
                    prob = stable_sigmoid(logit)
                    if prob >= self.threshold:
                        z[i, k] = 1

        return z
