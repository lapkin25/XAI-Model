import sys
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data
from sklearn.linear_model import LogisticRegression
import numpy as np
from sortedcontainers import SortedList
import matplotlib.pyplot as plt


def find_predictors_to_invert(data, predictors):
    # обучаем логистическую регрессию с выделенными признаками,
    #   выбираем признаки с отрицательными весами
    data.prepare(predictors, "Dead", [])
    logist_reg = LogisticRegression()
    logist_reg.fit(data.x, data.y)
    weights = logist_reg.coef_.ravel()
    invert_predictors = []
    for i, feature_name in enumerate(predictors):
        if weights[i] < 0:
            invert_predictors.append(feature_name)
    return invert_predictors


def find_threshold_rect(x_, y_, labels_):
    # сортируем точки по убыванию y
    ind = np.argsort(y_)
    ind = ind[::-1]
    x = x_[ind]
    y = y_[ind]
    labels = labels_[ind]
    n = x.size
    N1 = np.sum(labels)  # число единиц
    N0 = n - N1  # число нулей

    max_auc = 0.0
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

                auc = 0.5 + 0.5 * ones_right / N1 - 0.5 * zeros_right / N0

                if auc > max_auc:
                    max_auc = auc
                    a = x1
                    b = y[i]

    return a, b, max_auc


def plot_2d(x1, x1_name, x2, x2_name, y, a, b):
    min_x1 = np.min(x1)
    min_x2 = np.min(x2)
    max_x1 = np.max(x1)
    max_x2 = np.max(x2)

    plt.scatter(x1[y == 0], x2[y == 0], c='blue', linewidths=1)
    plt.scatter(x1[y == 1], x2[y == 1], c='red', linewidths=1)
    plt.axline((a, b), (a, max_x2), c='green')
    plt.axline((a, b), (max_x1, b), c='green')
    plt.xlabel(x1_name)
    plt.ylabel(x2_name)
    plt.show()


data = Data("DataSet.xlsx")
predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
invert_predictors = find_predictors_to_invert(data, predictors)
data.prepare(predictors, "Dead", invert_predictors)

ind1 = 0
ind2 = 5

a, b, auc = find_threshold_rect(data.x[:, ind1], data.x[:, ind2], data.y[:])
print(a, b)
print("AUC =", auc)
plot_2d(data.x[:, ind1], predictors[ind1], data.x[:, ind2], predictors[ind2], data.y[:], a, b)
