import sys
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as sklearn_metrics
import csv


# Функция возвращает:
#  px, py - координаты точки, через которую пройдет прямая
#  nx, ny - вектор нормали к прямой (смотрит в сторону класса "1")
def find_threshold_2d_slow(x1, x2, y):
    data_size = y.shape[0]
    N1 = np.sum(y)  # число "1"
    N0 = data_size - N1  # число "0"
    # координаты наилучшей прямой
    Px = None
    Py = None
    Nx = None
    Ny = None
    max_auc = 0.0
    for pivot_ind in range(data_size):
        if y[pivot_ind] == 0:
            continue
        #print(pivot_ind)
        for i in range(data_size):
            if y[i] == 0:
                # проводим прямую через точки pivot_ind и i
                px = x1[pivot_ind]
                py = x2[pivot_ind]
                mx = x1[i]
                my = x2[i]
                nx = my - py
                ny = px - mx
                # вычисляем AUC
                K1 = 0  # число "1" в положительном классе
                K0 = 0  # число "0" в положительном классе
                for j in range(data_size):
                    d = nx * (x1[j] - px) + ny * (x2[j] - py)
                    if d > 0:
                        # точка j прогнозируется как "1"
                        if y[j] == 1:
                            K1 += 1
                        else:
                            K0 += 1
                    else:
                        # точка j прогнозируется как "0"
                        pass
                auc = 0.5 + 0.5 * K1 / N1 - 0.5 * K0 / N0
                if auc > 0.5:
                    if auc > max_auc:
                        max_auc = auc
                        Px = px
                        Py = py
                        Nx = nx
                        Ny = ny
                else:  # auc <= 0.5
                    if 1 - auc > max_auc:
                        max_auc = 1 - auc
                        Px = px
                        Py = py
                        Nx = -nx
                        Ny = -ny

    #print(max_auc)
    return Px, Py, Nx / math.sqrt(Nx ** 2 + Ny ** 2), Ny / math.sqrt(Nx ** 2 + Ny ** 2)


# Функция возвращает:
#  px, py - координаты точки, через которую пройдет прямая
#  nx, ny - вектор нормали к прямой (смотрит в сторону класса "1")
def find_threshold_2d(x1, x2, y):
    data_size = y.shape[0]
    N1 = np.sum(y)  # число "1"
    N0 = data_size - N1  # число "0"
    # координаты наилучшей прямой
    Px = None
    Py = None
    Nx = None
    Ny = None
    max_auc = 0.0
    for pivot_ind in range(data_size):
        if y[pivot_ind] == 0:
            continue
        #print(pivot_ind)
        # считаем полярные углы всех точек
        a = np.zeros(data_size)
        a1 = np.zeros(data_size)
        for i in range(data_size):
            a1[i] = math.atan2(x2[i] - x2[pivot_ind], x1[i] - x1[pivot_ind])
            a[i] = a1[i]
            if a[i] < 0:
                a[i] += math.pi
        perm = np.argsort(a)
        x1_sorted = x1[perm]
        x2_sorted = x2[perm]
        y_sorted = y[perm]
        a1_sorted = a1[perm]
        a_sorted = a[perm]
        px = x1[pivot_ind]
        py = x2[pivot_ind]
        K1 = 0
        K0 = 0
        for i in range(data_size):
            if x2[i] >= x2[pivot_ind]:
                if y[i] == 1:
                    K1 += 1
                else:
                    K0 += 1
        for i in range(data_size):
            mx = x1_sorted[i]
            my = x2_sorted[i]
            if my >= py:  # точка исключается из класса "1"
                if y_sorted[i] == 1:
                    K1 -= 1
                else:
                    K0 -= 1
            else:  # точка включается в класс "1"
                if y_sorted[i] == 1:
                    K1 += 1
                else:
                    K0 += 1
            nx = -math.sin(a_sorted[i])
            ny = math.cos(a_sorted[i])
            # вычисляем AUC
            auc = 0.5 + 0.5 * K1 / N1 - 0.5 * K0 / N0
            if auc > 0.5:
                if auc > max_auc:
                    max_auc = auc
                    Px = px
                    Py = py
                    Nx = nx
                    Ny = ny
            else:  # auc <= 0.5
                if 1 - auc > max_auc:
                    max_auc = 1 - auc
                    Px = px
                    Py = py
                    Nx = -nx
                    Ny = -ny

    #print(max_auc)
    return Px, Py, Nx / math.sqrt(Nx ** 2 + Ny ** 2), Ny / math.sqrt(Nx ** 2 + Ny ** 2), max_auc


def plot_2d(x1, x1_name, x2, x2_name, y, p1, p2, n1, n2):
    min_x1 = np.min(x1)
    min_x2 = np.min(x2)
    max_x1 = np.max(x1)
    max_x2 = np.max(x2)
    # находим точки A, B, принадлежащие прямой
    Ax = min_x1
    Ay = ((n1 * p1 + n2 * p2) - n1 * min_x1) / n2
    Bx = max_x1
    By = ((n1 * p1 + n2 * p2) - n1 * max_x1) / n2

    plt.scatter(x1[y == 0], x2[y == 0], c='blue', linewidths=1)
    plt.scatter(x1[y == 1], x2[y == 1], c='red', linewidths=1)
    plt.axline((Ax, Ay), (Bx, By), c='green')
    plt.xlabel(x1_name)
    plt.ylabel(x2_name)
    plt.show()


class MaxAUC2Model:
    def __init__(self, num_combined_features):
        self.num_combined_features = num_combined_features
        self.features_used = None
        self.model = None
        self.thresholds = None

    def fit(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        z = None
        self.thresholds = []
        for ind1 in range(num_features):
            for ind2 in range(ind1 + 1, num_features):
                print("ind = ", ind1, "," , ind2)
                px, py, nx, ny, auc1 = find_threshold_2d(x[:, ind1], x[:, ind2], y[:])
                print(px, py, nx, ny)
                print("AUC =", auc1)
                row = np.zeros(data_size, dtype=int)
                for i in range(data_size):
                    if nx * (x[i, ind1] - px) + ny * (x[i, ind2] - py) >= 0:
                        row[i] = 1
                if z is None:
                    z = row
                else:
                    z = np.vstack((z, row))
                self.thresholds.append({'px': px, 'py': py, 'nx': nx, 'ny': ny, 'auc': auc1})

        z = z.T
        print(z)
        print(self.thresholds)

        # Отбор признаков методом включения
        max_auc = 0.0
        best_feature = None
        num_pair_features = z.shape[1]
        for feature in range(num_pair_features):
            print("Признак", feature + 1)
            model = LogisticRegression(solver='lbfgs', max_iter=10000)
            model.fit(z[:, feature].reshape(-1, 1), y)
            p = model.predict_proba(z[:, feature].reshape(-1, 1))[:, 1]
            auc = sklearn_metrics.roc_auc_score(y, p)
            if auc > max_auc:
                max_auc = auc
                best_feature = feature
        #print(best_feature, max_auc)

        features_used = [best_feature]

        for feature_cnt in range(1, self.num_combined_features):
            max_auc = 0.0
            best_feature = None
            for feature in range(num_pair_features):
                if feature not in features_used:
                    features_used.append(feature)
                    model = LogisticRegression(solver='lbfgs', max_iter=10000)
                    model.fit(z[:, features_used], y)
                    p = model.predict_proba(z[:, features_used])[:, 1]
                    auc = sklearn_metrics.roc_auc_score(y, p)
                    if auc > max_auc:
                        max_auc = auc
                        best_feature = feature
                    features_used.pop()
            features_used.append(best_feature)
            print("AUC =", max_auc)

        model = LogisticRegression(solver='lbfgs', max_iter=10000)
        model.fit(z[:, features_used], y)
        self.model = model
        self.features_used = features_used

    def predict_proba(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        z = None
        k = 0
        for ind1 in range(num_features):
            for ind2 in range(ind1 + 1, num_features):
                row = np.zeros(data_size, dtype=int)
                nx = self.thresholds[k]['nx']
                ny = self.thresholds[k]['ny']
                px = self.thresholds[k]['px']
                py = self.thresholds[k]['py']
                for i in range(data_size):
                    if nx * (x[i, ind1] - px) + ny * (x[i, ind2] - py) >= 0:
                        row[i] = 1
                if z is None:
                    z = row
                else:
                    z = np.vstack((z, row))
                k += 1

        z = z.T

        return self.model.predict_proba(z[:, self.features_used])


data = Data("DataSet.xlsx")
predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
data.prepare(predictors, "Dead", [], scale_data=False)  # не инвертируем предикторы

#ind1 = 0
#ind2 = 6

"""
px, py, nx, ny = find_threshold_2d_slow(data.x[:200, ind1], data.x[:200, ind2], data.y[:200])
print(px, py, nx, ny)
px, py, nx, ny = find_threshold_2d(data.x[:200, ind1], data.x[:200, ind2], data.y[:200])
print(px, py, nx, ny)
"""

#px, py, nx, ny = find_threshold_2d(data.x[:, ind1], data.x[:, ind2], data.y[:])
#print(px, py, nx, ny)
#plot_2d(data.x[:, ind1], predictors[ind1], data.x[:, ind2], predictors[ind2], data.y[:], px, py, nx, ny)


threshold = 0.03  #0.04
num_combined_features = 12  #10

num_splits = 50
random_state = 123

csvfile = open('splits.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
csvwriter.writerow(["auc", "sen", "spec"])

for it in range(1, 1 + num_splits):
    print("SPLIT #", it, "of", num_splits)

    x_train, x_test, y_train, y_test = \
        train_test_split(data.x, data.y, test_size=0.2, stratify=data.y)  #, random_state=random_state)  # закомментировать random_state

    max_auc_2_model = MaxAUC2Model(num_combined_features)
    max_auc_2_model.fit(x_train, y_train)

    print("Обучена модель")
    data_size, num_features = data.x.shape[0], data.x.shape[1]
    #print(max_auc_2_model.model.coef_.ravel())

    k = 0
    for ind1 in range(num_features):
        for ind2 in range(ind1 + 1, num_features):
            if k in max_auc_2_model.features_used:
                print(ind1, "(" + predictors[ind1] + ")", ind2, "(" + predictors[ind2] + ")",
                      max_auc_2_model.model.coef_[0][max_auc_2_model.features_used.index(k)])
                print(max_auc_2_model.thresholds[k])
                px = max_auc_2_model.thresholds[k]['px']
                py = max_auc_2_model.thresholds[k]['py']
                nx = max_auc_2_model.thresholds[k]['nx']
                ny = max_auc_2_model.thresholds[k]['ny']
                print("  Прямая", -nx / ny, "* x +", py + nx / ny * px)
                print("  AUC =", max_auc_2_model.thresholds[k]['auc'])
                #plot_2d(data.x[:, ind1], predictors[ind1], data.x[:, ind2], predictors[ind2], data.y[:], px, py, nx, ny)
            k += 1


    p = max_auc_2_model.predict_proba(x_test)[:, 1]
    auc = sklearn_metrics.roc_auc_score(y_test, p)
    print("AUC:", auc)
    # выводим качество модели
    y_pred = np.where(p > threshold, 1, 0)
    tn, fp, fn, tp = sklearn_metrics.confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print("Sens:", sensitivity, "Spec:", specificity)
    print("tp =", tp, "fn =", fn, "fp =", fp, "tn =", tn)

    csvwriter.writerow(map(str, [auc, sensitivity, specificity]))
