# взвешенная равномерная метрика
def weighted_max_distance(x1, x2, w):
    ans = 0
    for i in range(len(x1)):
        ans = max(ans, abs(w[i] * (x1[i] - x2[i])))
    return ans


# Возвращает для окрестности каждой точки
#   оценку отношения вероятности класса "1" к вероятности класса "0"
# Для оценки берется K ближайших точек в равномерной метрике с весами w[i]
def empirical_odds(x, y, K, w):
    data_size, num_features = x.shape[0], x.shape[1]
    odds = []
    for i in range(data_size):
        print(i)
        d = []
        for j in range(data_size):
            d.append(weighted_max_distance(x[i], x[j], w))
        points_decorated = [(d[j], j) for j in range(data_size)]
        points_decorated.sort()
        # индексы точек в порядке возрастания расстояния до i-й точки
        points_ind = [j for _, j in points_decorated]

        cnt0 = 0
        cnt1 = 0
        for k in range(K):
            if y[points_ind[k]] == 0:
                cnt0 += 1
            elif y[points_ind[k]] == 1:
                cnt1 += 1
            else:
                raise
        odds1 = cnt1 / cnt0
        odds.append(odds1)
    return odds



