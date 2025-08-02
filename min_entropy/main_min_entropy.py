import sys
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data
from sklearn.linear_model import LogisticRegression
import csv
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.metrics as sklearn_metrics
import math
#from permetrics.classification import ClassificationMetric


class MinEntropy:
    def __init__(self):
        self.cutoffs = None  # первичные пороги
        self.combined_features = None  # список троек (k, j, xj_cutoff) - вторичный порог
        #self.logistic_model = None
        #self.combined_weights = None
        #self.intercept = None

    # настраивает индивидуальные и комбинированные пороги,
    #   а также обучает логистическую регрессию на парных бинарных признаках
    def fit(self, x, y, simplified=False):
        data_size, num_features = x.shape[0], x.shape[1]
        self.cutoffs = np.zeros(num_features)
        # находим пороги в отдельности для каждого k-го признака
        for k in range(num_features):
            # разбиваем диапазон значений k-го признака на 100 частей
            grid = np.linspace(np.min(x[:, k]), np.max(x[:, k]), 100, endpoint=False)
            min_entropy = None
            optimal_cutoff = None
            for cutoff in grid:
                y_pred = np.where(x[:, k] >= cutoff, 1, 0).reshape(-1, 1)
                cm = confusion_matrix(y, y_pred)
                tp = cm[1, 1]
                tn = cm[0, 0]
                fp = cm[0, 1]
                fn = cm[1, 0]
                p1 = (tp + fn) / data_size  # доля реальных единиц в выборке
                p0 = (tn + fp) / data_size  # доля реальных нулей в выборке
                q1 = (tp + fp) / data_size  # доля предсказанных единиц
                q0 = (tn + fn) / data_size  # доля предсказанных нулей
                if p0 == 0 or p1 == 0 or q0 == 0 or q1 == 0:
                    continue
                entropy = -p1 * math.log(q1) - p0 * math.log(q0) + \
                    p1 * math.log(p1) + p0 * math.log(p0)
                if min_entropy is None or entropy < min_entropy:
                    min_entropy = entropy
                    optimal_cutoff = cutoff
            print("k =", k, " cutoff =", optimal_cutoff)
            self.cutoffs[k] = optimal_cutoff

        # находим вторичные пороги
        self.combined_features = []
        for k in range(num_features):
            filtering_k = x[:, k] >= self.cutoffs[k]
            for j in range(num_features):
                if k == j:
                    continue
                # разбиваем диапазон значений j-го признака на 100 частей
                grid = np.linspace(np.min(x[:, j]), np.max(x[:, j]), 100, endpoint=False)
                min_entropy = None
                optimal_cutoff = None
                for cutoff in grid:
                    xj_filtered = x[filtering_k, j]
                    y_filtered = y[filtering_k]
                    y_pred = np.where(xj_filtered >= cutoff, 1, 0).reshape(-1, 1)
                    cm = confusion_matrix(y_filtered, y_pred)
                    N = np.sum(filtering_k)
                    tp = cm[1, 1]
                    tn = cm[0, 0]
                    fp = cm[0, 1]
                    fn = cm[1, 0]
                    p1 = (tp + fn) / N  # доля реальных единиц в выборке
                    p0 = (tn + fp) / N  # доля реальных нулей в выборке
                    q1 = (tp + fp) / N  # доля предсказанных единиц
                    q0 = (tn + fn) / N  # доля предсказанных нулей
                    if p0 == 0 or p1 == 0 or q0 == 0 or q1 == 0:
                        continue

                    entropy = -p1 * math.log(q1) - p0 * math.log(q0) + \
                              p1 * math.log(p1) + p0 * math.log(p0)
                    if min_entropy is None or entropy < min_entropy:
                        min_entropy = entropy
                        optimal_cutoff = cutoff
                print("k =", k, "j = ", j, " cutoff =", optimal_cutoff)

                # в упрощенной модели берем вторичные пороги такие же, как первичные
                if simplified:
                    optimal_cutoff = self.cutoffs[j]

                self.combined_features.append((k, j, optimal_cutoff))

        """
        # формируем бинарные признаки
        z = None
        for k, j, xj_cutoff in self.combined_features:
            row = np.zeros(data_size, dtype=int)
            for i in range(data_size):
                if x[i, k] >= self.cutoffs[k] and x[i, j] >= xj_cutoff:
                    row[i] = 1
            if z is None:
                z = row
            else:
                z = np.vstack((z, row))
        z = z.T

        # обучаем логистическую регрессию
        model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)
        #model = LogisticRegression(max_iter=10000)
        model.fit(z, y)
        self.logistic_model = model
        """

    def predict_proba(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        probs = np.zeros(data_size)
        for i in range(data_size):
            for k, j, xj_cutoff in self.combined_features:
                # если правило сработало
                if x[i, k] >= self.cutoffs[k] and x[i, j] >= xj_cutoff:
                    probs[i] = 1
        return np.c_[1 - probs, probs]
        """
        z = None
        for k, j, xj_cutoff in self.combined_features:
            row = np.zeros(data_size, dtype=int)
            for i in range(data_size):
                if x[i, k] >= self.cutoffs[k] and x[i, j] >= xj_cutoff:
                    row[i] = 1
            if z is None:
                z = row
            else:
                z = np.vstack((z, row))
        z = z.T
        return self.logistic_model.predict_proba(z)
        """


def find_predictors_to_invert(data, predictors):
    # обучаем логистическую регрессию с выделенными признаками,
    #   выбираем признаки с отрицательными весами
    #data.prepare(predictors, "Dead", [])
    data.prepare(predictors, "isAFAfter", [])
    logist_reg = LogisticRegression()
    logist_reg.fit(data.x, data.y)
    weights = logist_reg.coef_.ravel()
    invert_predictors = []
    for i, feature_name in enumerate(predictors):
        if weights[i] < 0:
            invert_predictors.append(feature_name)
    return invert_predictors


def print_model(model, data, x_train, y_train, x_test, y_test):
    print("=" * 10 + "\nМодель")
    print("Пороги:")
    for k, feature_name in enumerate(predictors):
        val = data.get_coord(feature_name, model.cutoffs[k])
        s = '≤' if feature_name in data.inverted_predictors else '≥'
        print(feature_name, " ", s, val, sep='')
    print("Комбинированные пороги:")
    for k, j, xj_cutoff in model.combined_features:
        feature_name = predictors[k]
        val = data.get_coord(feature_name, model.cutoffs[k])
        s = '≤' if feature_name in data.inverted_predictors else '≥'
        print(feature_name, " ", s, val, sep='', end='')
        feature_name = predictors[j]
        val = data.get_coord(feature_name, xj_cutoff)
        s = '≤' if feature_name in data.inverted_predictors else '≥'
        print(" & ", feature_name, " ", s, val, sep='')

        tp = 0
        fp = 0
        #y_pred = np.zeros(x_train.shape[0])
        for i in range(x_train.shape[0]):
            if x_train[i, k] >= model.cutoffs[k] and x_train[i, j] >= xj_cutoff:
                #y_pred[i] = 1
                if y_train[i] == 1:
                    tp += 1
                else:
                    fp += 1
        print("На обучающей:   TP = ", tp, " FP = ", fp, " Prob = ", "inf" if tp == 0 and fp == 0 else tp / (tp + fp))

        tp = 0
        fp = 0
        for i in range(x_test.shape[0]):
            if x_test[i, k] >= model.cutoffs[k] and x_test[i, j] >= xj_cutoff:
                if y_test[i] == 1:
                    tp += 1
                else:
                    fp += 1
        print("На тестовой:   TP = ", tp, " FP = ", fp, " Prob = ", "inf" if tp == 0 and fp == 0 else tp / (tp + fp))

        #cm = ClassificationMetric(data.y, y_pred)
        #print("   Gini = ", cm.gini_index(average=None))
    #print("Веса:", model.individual_weights)
    #print("Комбинированные веса:", model.combined_weights)
    #print("Интерсепт:", model.intercept)


def t_model(model, x_test, y_test, p_threshold):
    p = model.predict_proba(x_test)[:, 1]
    auc = sklearn_metrics.roc_auc_score(y_test, p)
    print("AUC:", auc)
    # выводим качество модели
    y_pred = np.where(p > p_threshold, 1, 0)
    tn, fp, fn, tp = sklearn_metrics.confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print("Sens:", sensitivity, "Spec:", specificity)
    print("tp =", tp, "fn =", fn, "fp =", fp, "tn =", tn)
    return auc, sensitivity, specificity



#data = Data("DataSet.xlsx")
data = Data("STEMI.xlsx", STEMI=True)
#predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
predictors = ['Возраст', 'NER1', 'SIRI', 'СОЭ', 'TIMI после', 'СДЛА', 'Killip',
              'RR 600-1200', 'интервал PQ 120-200']
invert_predictors = find_predictors_to_invert(data, predictors)
print("Inverted:", invert_predictors)
predictors.append('RR 600-1200_')
invert_predictors.append('RR 600-1200_')
#data.prepare(predictors, "Dead", invert_predictors)
data.prepare(predictors, "isAFAfter", invert_predictors)

threshold = 0.12  #0.03

num_splits = 1
random_state = 1234  #123

csvfile = open('splits.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
csvwriter.writerow(["auc", "sen", "spec"])

csvwriter.writerow(["auc1", "sen1", "spec1", "auc2", "sen2", "spec2"])
for it in range(1, 1 + num_splits):
    print("SPLIT #", it, "of", num_splits)
    x_train, x_test, y_train, y_test = \
        train_test_split(data.x, data.y, test_size=0.2, stratify=data.y, random_state=123)  # закомментировать random_state

    model1 = MinEntropy()
    model1.fit(x_train, y_train, simplified=True)
    print_model(model1, data, x_train, y_train, x_test, y_test)
    auc1, sen1, spec1 = t_model(model1, x_test, y_test, threshold)

    model2 = MinEntropy()
    model2.fit(x_train, y_train, simplified=False)
    print_model(model2, data, x_train, y_train, x_test, y_test)
    auc2, sen2, spec2 = t_model(model2, x_test, y_test, threshold)

    csvwriter.writerow(map(str, [auc1, sen1, spec1, auc2, sen2, spec2]))