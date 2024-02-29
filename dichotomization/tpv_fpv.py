import numpy as np
from sortedcontainers import SortedList


# Находит пороги a, b, максимизирующие отношение числа единиц к числу нулей
#   в области {x >= a & y >= b}
#   при условии, что число нулей в этой области не меньше min_zero_count
# Возвращает числа a, b, max_rel
#   a, b - пороги для x и y
#   max_rel - максимум отношения числа единиц к числу нулей
def max_ones_zeros(x_, y_, labels_, min_zero_count, save_list=False):
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
    l = []

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
                        max_rel = rel
                        a = x1
                        b = y[i]
                    if save_list:
                        l.append((x1, y[i], rel))

    if not save_list:
        return a, b, max_rel
    else:
        return a, b, max_rel, l


# Находит пороги a, b, почти максимизирующие отношение числа единиц к числу нулей
#   в области {x >= a & y >= b}, чтобы минимизировать при этом порог a
#   и значение отношения числа единиц к числу нулей
#   отличалось от максимума менее чем на eps процентов
# Возвращает числа a, b, rel
#   a, b - пороги для x и y
#   rel - отношение числа единиц к числу нулей при этих порогах
def eps_max_ones_zeros_min_x(x_, y_, labels_, min_zero_count, eps):
    _, _, max_rel, l = max_ones_zeros(x_, y_, labels_, min_zero_count, save_list=True)
    opt_a = None
    opt_b = None
    opt_rel = None
    for a, b, rel in l:
        if opt_rel is None or rel > max_rel * (100 - eps) / 100 and a < opt_a:
            opt_a = a
            opt_b = b
            opt_rel = rel

    return opt_a, opt_b, opt_rel


# Находит пороги a, b, почти максимизирующие отношение числа единиц к числу нулей
#   в области {x >= a & y >= b}, чтобы максимизировать при этом порог b
#   и значение отношения числа единиц к числу нулей
#   отличалось от максимума менее чем на eps процентов
# Возвращает числа a, b, rel
#   a, b - пороги для x и y
#   rel - отношение числа единиц к числу нулей при этих порогах
def eps_max_ones_zeros_max_y(x_, y_, labels_, min_zero_count, eps):
    _, _, max_rel, l = max_ones_zeros(x_, y_, labels_, min_zero_count, save_list=True)
    opt_a = None
    opt_b = None
    opt_rel = None
    for a, b, rel in l:
        if opt_rel is None or rel > max_rel * (100 - eps) / 100 and b > opt_b:
            opt_a = a
            opt_b = b
            opt_rel = rel

    return opt_a, opt_b, opt_rel


# Находит пороги a, b, почти максимизирующие отношение числа единиц к числу нулей
#   в области {x >= a & y >= b}, чтобы минимизировать при этом порог b
#   и значение отношения числа единиц к числу нулей
#   отличалось от максимума менее чем на eps процентов
# Возвращает числа a, b, rel
#   a, b - пороги для x и y
#   rel - отношение числа единиц к числу нулей при этих порогах
def eps_max_ones_zeros_min_y(x_, y_, labels_, min_zero_count, eps):
    _, _, max_rel, l = max_ones_zeros(x_, y_, labels_, min_zero_count, save_list=True)
    opt_a = None
    opt_b = None
    opt_rel = None
    for a, b, rel in l:
        if opt_rel is None or rel > max_rel * (100 - eps) / 100 and b < opt_b:
            opt_a = a
            opt_b = b
            opt_rel = rel

    return opt_a, opt_b, opt_rel
