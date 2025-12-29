import sys

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data
from sklearn.linear_model import LogisticRegression
import csv
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.metrics as sklearn_metrics
import math
#from pyDOE import lhs
#from geneticalgorithm import geneticalgorithm as ga
#from permetrics.classification import ClassificationMetric
from sklearn.naive_bayes import BernoulliNB
import itertools
from sympy.logic import SOPform
from sympy import symbols
import pygad
import random

DEFAULT_PROB = 1e-8

class BinaryProbabilityModel:
    def __init__(self):
        self.prob = None
        self.num_features = None

    def bin_code(self, x):
        """
        Найти двоичный код ортанта
        """
        num_features = len(x)
        ans = 0
        z = 1
        for i in range(num_features):
            ans += z * x[i]
            z *= 2
        return ans

    def fit(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        self.num_features = num_features
        # считаем точки во всех ортантах...
        N1 = np.zeros(2 ** num_features, dtype=int)  # число "1" в ортантах
        N0 = np.zeros(2 ** num_features, dtype=int)  # число "0" в ортантах
        total_N1 = np.sum(y)
        total_N0 = data_size - total_N1
        for k in range(data_size):
            code = self.bin_code(x[k, :])
            if y[k] == 1:
                N1[code] += 1
            else:
                N0[code] += 1
        # считаем вероятности, объединяя соседние ортанты...
        sum_N1 = np.zeros(2 ** num_features, dtype=int)
        sum_N0 = np.zeros(2 ** num_features, dtype=int)
        self.prob = np.zeros(2 ** num_features)  # оценка вероятности в ортанте
        FIX_FEATURES = 6  #7
        for u in itertools.product([0, 1], repeat=num_features):
            # u - это некоторый двоичный код
            code_u = self.bin_code(u)
            if N0[code_u] == 0 and N1[code_u] == 0:
                continue
            # выбираем все возможные семерки предикторов
            for fixed_indices in itertools.combinations(range(num_features), FIX_FEATURES):
                # фиксируем эти 7 значений, остальные меняем произвольным образом
                for v in itertools.product([0, 1], repeat=num_features - FIX_FEATURES):
                    # формируем двоичный код из фиксированных и не фиксированных индексов
                    w = np.array(u)
                    j = 0
                    for i in range(num_features):
                        if i not in fixed_indices:
                            w[i] = v[j]
                            j += 1
                    code = self.bin_code(w)
                    sum_N1[code] += N1[code_u]
                    sum_N0[code] += N0[code_u]
        for code in range(2 ** num_features):
            if sum_N1[code] + sum_N0[code] != 0:  # непонятно, что с этим делать
                self.prob[code] = sum_N1[code] / (sum_N1[code] + sum_N0[code])
            else:
                self.prob[code] = DEFAULT_PROB

    def predict_proba(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        p = np.zeros(data_size)
        for k in range(data_size):
            code = self.bin_code(x[k, :])
            p[k] = self.prob[code]
        return np.c_[1 - p, p]

    # выделить решающие правила
    def interpret(self, threshold):
        predictors1 = list(map(lambda s: "_".join(s.split()), predictors))
        vars = symbols(" ".join(predictors1))
        minterms = []
        dontcares = []
        print('Переменные:', vars)
        for v in itertools.product([0, 1], repeat=self.num_features):
            code = self.bin_code(v)
            if self.prob[code] == DEFAULT_PROB:
                dontcares.append(v)
            if self.prob[code] > threshold:
                minterms.append(v)
            #print(v, self.prob[code])
        print('dontcares =', dontcares)
        print('minterms =', minterms)
        dnf = SOPform(vars, minterms, dontcares)
        print(dnf)  # вывод сокращенной ДНФ
        conjs = []  # список конъюнктов (список списков имен переменных)
        for mt in str(dnf).split("|"):
            v = list(map(lambda s: s.strip(), mt.split("&")))
            # убираем скобки с начала и с конца
            v[0] = v[0][1:]
            v[-1] = v[-1][:-1]
            # print(v)
            # assert(len(v) >= 5)
            if len(v) <= 5:
                conjs.append(v)


class ProbabilityMinEntropyModel:
    def __init__(self):
        self.cutoffs = None  # пороги для каждого предиктора
        self.clf = None

    # находит оптимальные пороги
    def fit(self, x, y, set_cutoffs=None):
        data_size, num_features = x.shape[0], x.shape[1]
        self.num_features = num_features
        self.cutoffs = np.zeros(num_features)

        lb = np.min(x, axis=0)
        ub = np.max(x, axis=0)

        def calc_J(c):
            z = np.zeros((data_size, num_features), dtype=int)
            for j in range(num_features):
                z[:, j] = np.where(x[:, j] >= c[j], 1, 0)
            clf = BinaryProbabilityModel()
            clf.fit(z, y)
            yp = clf.predict_proba(z)
            return sklearn_metrics.log_loss(y, yp)

        def fitness_func(ga_instance, solution, solution_idx):
            log_loss = calc_J(solution)
            return -log_loss

        num_generations = 30  #100  # Number of generations.
        num_parents_mating = 10  # Number of solutions to be selected as parents in the mating pool.

        sol_per_pop = 20  # Number of solutions in the population.

        num_genes = num_features

        gene_space = [{'low': lb[j], 'high': ub[j]} if predictors[j] != "Killip class" else [1.6] for j in range(num_features)]

        def on_generation(ga_instance):
            print(f"Generation = {ga_instance.generations_completed}")
            print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")

        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               fitness_func=fitness_func,
                               on_generation=on_generation,
                               gene_space=gene_space)

        if set_cutoffs is None:
            # Running the GA to optimize the parameters of the function.
            ga_instance.run()

            #ga_instance.plot_fitness()

            # Returning the details of the best solution.
            solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
            print(f"Parameters of the best solution : {solution}")
            print(f"Fitness value of the best solution = {solution_fitness}")

            self.cutoffs = solution
        else:
            self.cutoffs = set_cutoffs

        z = np.zeros((data_size, num_features), dtype=int)
        for j in range(num_features):
            z[:, j] = np.where(x[:, j] >= self.cutoffs[j], 1, 0)
        clf = BinaryProbabilityModel()
        clf.fit(z, y)
        self.clf = clf

    def predict_proba(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        z = np.zeros((data_size, num_features), dtype=int)
        for j in range(num_features):
            z[:, j] = np.where(x[:, j] >= self.cutoffs[j], 1, 0)
        return self.clf.predict_proba(z)


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



data = Data("DataSet.xlsx")

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]

#predictors = ["Age", "Killip class", "Cr", "EF LV", "NEUT"]

"""
threshold = 0.15:
    (Age & Cr & EF_LV & Glu & NEUT) | (Age & Cr & EF_LV & Killip_class & NEUT) | (Age & Cr & Glu & HR & NEUT) | (Age & Cr & Glu & Killip_class & NEUT) | (Age & Cr & Glu & NEUT & SBP) | (Cr & EF_LV & EOS & Killip_class & SBP) | (Cr & EF_LV & EOS & NEUT & SBP) | (Cr & EOS & HR & Killip_class & SBP) | (EF_LV & HR & Killip_class & NEUT & SBP) 
"""

"""
predictors = ["Age", "Cr", "EF LV", "NEUT", "Glu"]
Age ≥69.46343099746149
Cr ≥167.944027130994
EF LV ≤33.033174188660354
NEUT ≥75.42117692704959
Glu ≥7.062776779828779
EF_LV | (Age & Cr) | (Age & NEUT) | (Cr & Glu) | (Cr & NEUT) | (Glu & NEUT)

predictors = ["Age", "Killip class", "Cr", "EF LV", "NEUT"]
Age ≥70.57561758066473
Killip class ≥2.9698577158827306
Cr ≥154.18287069093304
EF LV ≤35.37571892147048
NEUT ≥75.21380169168862
(Age & NEUT) | (Cr & NEUT) | (EF_LV & NEUT) | (Killip_class & NEUT) | (Age & Cr & ~Killip_class) | (Age & EF_LV & ~Killip_class) | (Age & Killip_class & ~Cr) | (Cr & EF_LV & ~Age) | (Cr & Killip_class & ~Age) | (EF_LV & Killip_class & ~Age)

predictors = ["Age", "HR", "Cr", "NEUT", "Glu"]
Age ≥68.40573101284396
HR ≥84.51792574832406
Cr ≥159.83739760787995
NEUT ≥74.99352559492017
Glu ≥12.605976001915433
(Age & NEUT) | (Cr & NEUT) | (Glu & NEUT) | (HR & NEUT) | (Cr & Glu & ~HR) | (Cr & HR & ~Glu) | (Glu & HR & ~Age & ~Cr)
"""

#set_cutoffs = None

set_cutoffs = [70, 82, 3, 135.7, 45, 75.6, 0.481, 0.24, 6.83, 115]

invert_predictors = find_predictors_to_invert(data, predictors)
data.prepare(predictors, "Dead", invert_predictors)

if set_cutoffs is not None:
    transformed_cutoffs = []
    for i, nt in enumerate(set_cutoffs):
        val_normal = nt
        if predictors[i] in invert_predictors:
            val_normal = -val_normal
        val = (val_normal - data.scaler_mean[i]) / data.scaler_scale[i]
        transformed_cutoffs.append(val)
    set_cutoffs = transformed_cutoffs


threshold = 0.05

num_splits = 10
random_state = 123

np.random.seed(random_state)
random.seed(random_state)

csvfile = open('splits.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')

csvwriter.writerow(["auc1", "sen1", "spec1"])
for it in range(1, 1 + num_splits):
    print("SPLIT #", it, "of", num_splits)
    x_train_all, x_test_all, y_train_all, y_test_all = \
        train_test_split(data.x, data.y, test_size=0.2, stratify=data.y)  #, random_state=random_state)  # закомментировать random_state


    if set_cutoffs is None:
        skf = StratifiedKFold(n_splits=5)  #8)
        all_cutpoints = []
        #auc_history = []
        for fold, (train_index, test_index) in enumerate(skf.split(x_train_all, y_train_all)):
            x_train, x_test = data.x[train_index, :], data.x[test_index, :]
            y_train, y_test = data.y[train_index], data.y[test_index]
            print("  Fold", fold + 1)

            model1 = ProbabilityMinEntropyModel()
            model1.fit(x_train, y_train)
            print_model(model1, data)
            all_cutpoints.append(model1.cutoffs)

            t_model(model1, x_test, y_test, threshold)
            #model1.interpret()

    if set_cutoffs is None:
        # усредняем найденные пороги
        cutpoints = np.mean(np.vstack(all_cutpoints), axis=0)
    else:
        cutpoints = set_cutoffs
    avg_model = ProbabilityMinEntropyModel()
    avg_model.fit(x_train_all, y_train_all, set_cutoffs=cutpoints)
    print_model(avg_model, data)
    avg_model.clf.interpret(threshold)
    auc1, sen1, spec1 = t_model(avg_model, x_test_all, y_test_all, threshold)
    #avg_model.interpret()

    csvwriter.writerow(map(str, [auc1, sen1, spec1]))
