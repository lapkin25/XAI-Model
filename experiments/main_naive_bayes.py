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
from pyDOE import lhs
from geneticalgorithm import geneticalgorithm as ga
#from permetrics.classification import ClassificationMetric
from sklearn.naive_bayes import BernoulliNB

class NaiveBayes:
    def __init__(self):
        self.cutoffs = None  # пороги для каждого предиктора

    # находит оптимальные пороги
    def fit(self, x, y, use_genetic=True):
        data_size, num_features = x.shape[0], x.shape[1]
        self.cutoffs = np.zeros(num_features)

        num_samples = 10000

        np.random.seed(1234)
        lb = np.min(x, axis=0)
        ub = np.max(x, axis=0)

        def calc_J(c):
            z = np.zeros((data_size, num_features), dtype=int)
            for j in range(num_features):
                z[:, j] = np.where(x[:, j] >= c[j], 1, 0)
            clf = BernoulliNB()
            clf.fit(z, y)
            y_pred = clf.predict(z)
            fpr, tpr, _ = sklearn_metrics.roc_curve(y, y_pred)
            return sklearn_metrics.auc(fpr, tpr)

        def calc_J_old(c):
            N1 = np.sum(y)
            N0 = data_size - N1
            P1 = np.zeros(num_features)  # P(X_j >= c_j / Y = 1)
            P0 = np.zeros(num_features)  # P(X_j >= c_j / Y = 0)
            for j in range(num_features):
                cnt1 = 0
                cnt0 = 0
                for i in range(data_size):
                    if x[i, j] >= c[j]:
                        if y[i] == 1:
                            cnt1 += 1
                        else:
                            cnt0 += 1
                P1[j] = cnt1 / N1
                P0[j] = cnt0 / N0

            tp = 0
            fp = 0
            fn = 0
            tn = 0
            for i in range(data_size):
                M1 = 1.0 * N1
                M0 = 1.0 * N0
                for j in range(num_features):
                    F_j = x[i, j] >= c[j]
                    if F_j:
                        M1 *= P1[j]
                        M0 *= P0[j]
                    else:
                        M1 *= 1 - P1[j]
                        M0 *= 1 - P0[j]
                if M1 > M0:
                    yp = 1
                else:
                    yp = 0
                if yp == 1 and y[i] == 1:
                    tp += 1
                elif yp == 1 and y[i] == 0:
                    fp += 1
                elif yp == 0 and y[i] == 1:
                    fn += 1
                else:
                    tn += 1
            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            return 0.5 * (sens + spec)


        if use_genetic:
            varbound = np.array([[lb[j], ub[j]] for j in range(num_features)])
            print(varbound)

            def f(c):
                return -calc_J(c)

            model = ga(function=f, dimension=num_features, variable_type='real', variable_boundaries=varbound)
            result = model.run()
            self.cutoffs = result['variable']
        else:
            samples = lb + (ub - lb) * lhs(num_features, num_samples)

            max_J = 0.0
            best_cutoffs = None
            for si in range(num_samples):
                J = calc_J(samples[si, :])
                if J > max_J:
                    max_J = J
                    best_cutoffs = samples[si, :]

            print("max_J =", max_J)
            print("best_cutoffs =", best_cutoffs)
            self.cutoffs = best_cutoffs[:]

    def predict(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        # TODO: реализовать

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


def print_model(model, data):
    print("=" * 10 + "\nМодель")
    print("Пороги:")
    for k, feature_name in enumerate(predictors):
        val = data.get_coord(feature_name, model.cutoffs[k])
        s = '≤' if feature_name in data.inverted_predictors else '≥'
        print(feature_name, " ", s, val, sep='')


def test_model(model, x_test, y_test, p_threshold):
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



data = Data("DataSet.xlsx")
predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
invert_predictors = find_predictors_to_invert(data, predictors)
data.prepare(predictors, "Dead", invert_predictors)

threshold = 0.03

num_splits = 1
random_state = 123

csvfile = open('splits.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
csvwriter.writerow(["auc", "sen", "spec"])

csvwriter.writerow(["auc1", "sen1", "spec1", "auc2", "sen2", "spec2"])
for it in range(1, 1 + num_splits):
    print("SPLIT #", it, "of", num_splits)
    x_train, x_test, y_train, y_test = \
        train_test_split(data.x, data.y, test_size=0.2, stratify=data.y, random_state=123)  # закомментировать random_state

    model1 = NaiveBayes()
    model1.fit(x_train, y_train)
    print_model(model1, data)
    auc1, sen1, spec1 = test_model(model1, x_test, y_test, threshold)

    csvwriter.writerow(map(str, [auc1, sen1, spec1]))
