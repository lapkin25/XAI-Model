import numpy as np
from sortedcontainers import SortedList


# Находит пороги a, b
#   (в окрестности порогов a0, b0: |a - a0| <= da, |b - b0| <= db),
#   максимизирующие отношение числа единиц к числу нулей
#   в области {x >= a & y >= b}
#   при условии, что число нулей в этой области не меньше min_zero_count
# Возвращает числа a, b, max_rel
#   a, b - пороги для x и y
#   max_rel - максимум отношения числа единиц к числу нулей
def new_max_ones_zeros(x_, y_, labels_, min_zero_count, a0, b0, da, db):
    # сортируем точки по убыванию y
    ind = np.argsort(y_)
    ind = ind[::-1]
    x = x_[ind]
    y = y_[ind]
    labels = labels_[ind]
    n = x.size

    max_rel = None
    a = None
    b = None

    x_coord = SortedList()  # x-координаты всех добавленных точек
    ones_x_coord = SortedList()  # x-координаты всех добавленных единиц
    for i in range(n):
        x_coord.add(x[i])
        if labels[i] == 1:
            ones_x_coord.add(x[i])
            for j, x1 in enumerate(reversed(ones_x_coord)):
                # ones_right - число единиц не левее x1
                ones_right = j + 1
                # points_right - число точек не левее x1
                points_right = (i + 1) - x_coord.index(x1)
                # zeros_right - число нулей не левее x1
                zeros_right = points_right - ones_right
                if zeros_right >= min_zero_count:
                    rel = ones_right / zeros_right
                    if max_rel is None or rel > max_rel:
                        if a0 is None or abs(x1 - a0) <= da and abs(y[i] - b0) <= db:
                            max_rel = rel
                            a = x1
                            b = y[i]

    return a, b, max_rel
