import numpy as np
from blist import sortedlist

# Находит пороги a, b, максимизирующие отношение числа единиц к числу нулей
#   в области {x >= a & y >= b}
#   при условии, что число нулей в этой области не меньше min_zero_count
# Возвращает числа a, b, max_rel
#   a, b - пороги для x и y
#   max_rel - максимум отношения числа единиц к числу нулей
def max_ones_zeros(x_, y_, labels_, min_zero_count):
    # сортируем точки по убыванию y
    ind = np.argsort(y_)
    ind = ind[::-1]
    x = x_[ind]
    y = y_[ind]
    labels = labels_[ind]
    n = x.size

    x_coord = sortedlist()  # x-координаты всех добавленных точек
    ones_x_coord = sortedlist()  # x-координаты всех добавленных единиц
    for i in range(n):
        x_coord.add(x[i])
        if labels[i] == 1:
            ones_x_coord.add(x[i])
        for ones_right, x1 in enumerate(reversed(ones_x_coord)):
            # ones_right - число единиц не левее x1
            # points_right - число точек не левее x1
            points_right = x_coord.index()



