import sys
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as sklearn_metrics
from sklearn.model_selection import train_test_split
import numpy as np
from sortedcontainers import SortedList
import matplotlib.pyplot as plt
import csv
import math


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

    min_entropy = None
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

                zeros_cnt = (len(x_coord) - len(ones_x_coord))

                tp = ones_right
                tn = (N0 - zeros_cnt) + (zeros_cnt - zeros_right)
                fp = zeros_right
                fn = n - (tp + tn + fp)
                p1 = (tp + fn) / n  # доля реальных единиц в выборке
                p0 = (tn + fp) / n  # доля реальных нулей в выборке
                q1 = (tp + fp) / n  # доля предсказанных единиц
                q0 = (tn + fn) / n  # доля предсказанных нулей
                if p0 == 0 or p1 == 0 or q0 == 0 or q1 == 0:
                    continue

                entropy = -p1 * math.log(q1) - p0 * math.log(q0) + \
                    p1 * math.log(p1) + p0 * math.log(p0)

                if min_entropy is None or entropy < min_entropy:
                    min_entropy = entropy
                    a = x1
                    b = y[i]

    return a, b


def plot_2d(x1, x1_name, x2, x2_name, y, a, b):
    min_x1 = np.min(x1)
    min_x2 = np.min(x2)
    max_x1 = np.max(x1)
    max_x2 = np.max(x2)

    val_x1 = np.array([data.get_coord(x1_name, x1[i]) for i in range(x1.shape[0])])
    val_x2 = np.array([data.get_coord(x2_name, x2[i]) for i in range(x2.shape[0])])
    val_a = data.get_coord(x1_name, a)
    val_b = data.get_coord(x2_name, b)

    plt.scatter(val_x1[y == 0], val_x2[y == 0], c='blue', linewidths=1)
    plt.scatter(val_x1[y == 1], val_x2[y == 1], c='red', linewidths=1)
    plt.axline((val_a, val_b), (val_a, max(val_x2)), c='green')
    plt.axline((val_a, val_b), (max(val_x1), val_b), c='green')
    plt.xlabel(x1_name)
    plt.ylabel(x2_name)
    plt.show()


class MinEntropy2Model:
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
                a, b = find_threshold_rect(x[:, ind1], x[:, ind2], y[:])
                print(a, b)
                #print("AUC =", auc1)
                row = np.zeros(data_size, dtype=int)
                for i in range(data_size):
                    if x[i, ind1] >= a and x[i, ind2] >= b:
                        row[i] = 1
                if z is None:
                    z = row
                else:
                    z = np.vstack((z, row))
                self.thresholds.append({'a': a, 'b': b})

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
                a = self.thresholds[k]['a']
                b = self.thresholds[k]['b']
                for i in range(data_size):
                    if x[i, ind1] >= a and x[i, ind2] >= b:
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
invert_predictors = find_predictors_to_invert(data, predictors)
data.prepare(predictors, "Dead", invert_predictors)

"""
ind1 = 0
ind2 = 5

a, b, auc = find_threshold_rect(data.x[:, ind1], data.x[:, ind2], data.y[:])
print(a, b)
print("AUC =", auc)
plot_2d(data.x[:, ind1], predictors[ind1], data.x[:, ind2], predictors[ind2], data.y[:], a, b)
"""

threshold = 0.03  #0.04
num_combined_features = 12  #10

num_splits = 1  #50
random_state = 123

csvfile = open('splits.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
csvwriter.writerow(["auc", "sen", "spec"])

for it in range(1, 1 + num_splits):
    print("SPLIT #", it, "of", num_splits)

    x_train, x_test, y_train, y_test = \
        train_test_split(data.x, data.y, test_size=0.2, stratify=data.y, random_state=random_state)  # закомментировать random_state

    min_entropy_model = MinEntropy2Model(num_combined_features)
    min_entropy_model.fit(x_train, y_train)

    print("Обучена модель")
    data_size, num_features = data.x.shape[0], data.x.shape[1]
    #print(max_auc_2_model.model.coef_.ravel())

    k = 0
    for ind1 in range(num_features):
        for ind2 in range(ind1 + 1, num_features):
            if k in min_entropy_model.features_used:
                print(ind1, ind2, min_entropy_model.model.coef_[0][min_entropy_model.features_used.index(k)])
                print(min_entropy_model.thresholds[k])
                a = min_entropy_model.thresholds[k]['a']
                b = min_entropy_model.thresholds[k]['b']

                feature1 = predictors[ind1]
                feature2 = predictors[ind2]
                val_a = data.get_coord(feature1, a)
                val_b = data.get_coord(feature2, b)
                s1 = '≤' if feature1 in data.inverted_predictors else '≥'
                s2 = '≤' if feature2 in data.inverted_predictors else '≥'
                print('  ', feature1, " ", s1, val_a, sep='')
                print('  ', feature2, " ", s2, val_b, sep='')
                #print("  AUC =", max_auc_rect_model.thresholds[k]['auc'])

                # вывод графика
                plot_2d(data.x[:, ind1], predictors[ind1], data.x[:, ind2], predictors[ind2], data.y[:], a, b)

            k += 1


    p = min_entropy_model.predict_proba(x_test)[:, 1]
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
