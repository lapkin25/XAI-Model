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
import itertools
from sympy.logic import SOPform
from sympy import symbols


class NaiveBayes:
    def __init__(self):
        self.cutoffs = None  # пороги для каждого предиктора
        self.clf = None
        self.num_features = None
        self.triples = None

    # находит оптимальные пороги
    def fit(self, x, y, use_genetic=True):
        data_size, num_features = x.shape[0], x.shape[1]
        self.num_features = num_features
        self.cutoffs = np.zeros(num_features)
        self.triples = None
        self.triples_logistic_model = None

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
            #print(varbound)

            algorithm_param = {'max_num_iteration': 1000, \
                               'population_size': 100, \
                               'mutation_probability': 0.1, \
                               'elit_ratio': 0.01, \
                               'crossover_probability': 0.5, \
                               'parents_portion': 0.3, \
                               'crossover_type': 'uniform', \
                               'max_iteration_without_improv': None, \
                               'convergence_curve': False}

            def f(c):
                return -calc_J(c)

            model = ga(function=f, dimension=num_features, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
            model.run()
            self.cutoffs = model.output_dict['variable']
            #self.cutoffs = np.array([0.72358296, 2.20297099, 1.20732979, 0.71223686, 0.9679637, 0.55899361, 0.30359288, 5.04075139, 0.70719669, 1.0690253]) # result['variable']
            z = np.zeros((data_size, num_features), dtype=int)
            for j in range(num_features):
                z[:, j] = np.where(x[:, j] >= self.cutoffs[j], 1, 0)
            clf = BernoulliNB()
            clf.fit(z, y)
            self.clf = clf
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
        z = np.zeros((data_size, num_features), dtype=int)
        for j in range(num_features):
            z[:, j] = np.where(x[:, j] >= self.cutoffs[j], 1, 0)
        y_pred = self.clf.predict(z)
        return y_pred

    def predict_proba(self, x):
        yp = self.predict(x)
        return np.c_[1 - yp, yp]

    # выделить решающие правила
    def interpret(self):
        predictors1 = list(map(lambda s: "_".join(s.split()), predictors))
        vars = symbols(" ".join(predictors1))
        minterms = []
        print(vars)
        for v in itertools.product([0, 1], repeat=self.num_features):
            #print(np.array(v).reshape(1, -1))
            y_pred = self.clf.predict(np.array(v).reshape(1, -1))
            print(v, '->', y_pred)
            if y_pred:
                minterms.append(v)
        dnf = SOPform(vars, minterms)
        print(dnf)  # вывод сокращенной ДНФ
        conjs = []  # список конъюнктов (список списков имен переменных)
        for mt in str(dnf).split("|"):
            v = list(map(lambda s: s.strip(), mt.split("&")))
            # убираем скобки с начала и с конца
            v[0] = v[0][1:]
            v[-1] = v[-1][:-1]
            #print(v)
            assert(len(v) >= 5)
            if len(v) == 5:
                conjs.append(v)
        print(conjs)
        print(predictors1)
        triples = []
        for v in itertools.combinations(predictors1, 3):
            #print(v)
            cnt = 0
            for c in conjs:
                if all([vitem in c for vitem in v]):
                    cnt += 1
            #print(v, cnt)
            triples.append((v, cnt))
        triples.sort(key=lambda t: t[1], reverse=True)
        for t in triples:
            print(t[0], t[1])

        self.triples = []
        for t in triples:
            if t[1] >= 20:
                self.triples.append(list(map(lambda a: predictors1.index(a), t[0])))


    def fit_with_triples(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        num_triples = len(self.triples)
        z = np.zeros((data_size, num_triples), dtype=int)
        for i in range(data_size):
            for j in range(num_triples):
                ind1, ind2, ind3 = self.triples[j]
                z[i, j] = (x[i, ind1] >= self.cutoffs[ind1]) \
                    and (x[i, ind2] >= self.cutoffs[ind2]) \
                    and (x[i, ind3] >= self.cutoffs[ind3])
        model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)
        model.fit(z, y)
        self.triples_logistic_model = model

    def predict_proba_with_triples(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        num_triples = len(self.triples)
        z = np.zeros((data_size, num_triples), dtype=int)
        for i in range(data_size):
            for j in range(num_triples):
                ind1, ind2, ind3 = self.triples[j]
                z[i, j] = (x[i, ind1] >= self.cutoffs[ind1]) \
                    and (x[i, ind2] >= self.cutoffs[ind2]) \
                    and (x[i, ind3] >= self.cutoffs[ind3])
        return self.triples_logistic_model.predict_proba(z)


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

num_splits = 10
random_state = 123

csvfile = open('splits.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
csvwriter.writerow(["auc", "sen", "spec"])

csvwriter.writerow(["auc1", "sen1", "spec1", "auc2", "sen2", "spec2"])
for it in range(1, 1 + num_splits):
    print("SPLIT #", it, "of", num_splits)
    x_train, x_test, y_train, y_test = \
        train_test_split(data.x, data.y, test_size=0.2, stratify=data.y)   #, random_state=123)  # закомментировать random_state

    model1 = NaiveBayes()
    model1.fit(x_train, y_train)
    print_model(model1, data)
    auc1, sen1, spec1 = test_model(model1, x_test, y_test, threshold)
    model1.interpret()
    #print(model1.triples)
    model1.fit_with_triples(x_train, y_train)
    for i, t in enumerate(model1.triples):
        print(predictors[t[0]], '&', predictors[t[1]], '&', predictors[t[2]], '-> w =', model1.triples_logistic_model.coef_[0][i])
    p = model1.predict_proba_with_triples(x_test)[:, 1]
    auc = sklearn_metrics.roc_auc_score(y_test, p)
    print("AUC:", auc)
    # выводим качество модели
    y_pred = np.where(p > threshold, 1, 0)
    tn, fp, fn, tp = sklearn_metrics.confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print("Sens:", sensitivity, "Spec:", specificity)
    print("tp =", tp, "fn =", fn, "fp =", fp, "tn =", tn)
    auc2 = auc
    sen2 = sensitivity
    spec2 = specificity

    csvwriter.writerow(map(str, [auc1, sen1, spec1, auc2, sen2, spec2]))
