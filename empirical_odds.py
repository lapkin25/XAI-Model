import math
import numpy as np
from tqdm import tqdm

# взвешенная равномерная метрика
def weighted_max_distance(x1, x2, w):
    ans = 0
    num_features = len(x1)
    for i in range(num_features):
        ans = max(ans, abs(w[i] * (x1[i] - x2[i])))
    return ans


# Возвращает для окрестности каждой точки
#   оценку отношения вероятности класса "1" к вероятности класса "0"
# Для оценки берется K ближайших точек в равномерной метрике с весами w[i]
def empirical_log_odds(x, y, K, w):
    data_size, num_features = x.shape[0], x.shape[1]
    log_odds = []
    for i in tqdm(range(data_size)):
        d = np.array([0 for _ in range(data_size)])
        for j in range(data_size):
            d[j] = weighted_max_distance(x[i], x[j], w)
        # индексы точек в порядке возрастания расстояния до i-й точки
        points_ind = d.argsort()

        cnt0 = 0
        cnt1 = 0
        for k in range(K):
            if y[points_ind[k]] == 0:
                cnt0 += 1
            elif y[points_ind[k]] == 1:
                cnt1 += 1
            else:
                raise
        log_odds1 = math.log((cnt1 / K + 0.5 / K) / (cnt0 / K + 0.5 / K))
        log_odds.append(log_odds1)
    return log_odds



