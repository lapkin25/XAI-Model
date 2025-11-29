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


class BooleanClassifier:
    def __init__(self):
        self.N1 = None
        self.N0 = None
        self.total_N0 = None
        self.total_N1 = None

    def bin_code(self, x):
        """
        Найти двоичный код кластера
        """
        num_features = len(x)
        ans = 0
        z = 1
        for i in range(num_features):
            ans += z * x[i]
            z *= 2
        return ans

    def fit(self, x, y):
        # TODO: задать частичную булеву функцию и минимизировать ДНФ
        # minterms = ...
        # dontcares = ...

        data_size, num_features = x.shape[0], x.shape[1]

        #y_pred = np.zeros_like(y)



        self.N1 = np.zeros(2 ** num_features, dtype=int)  # число "1" в кластерах
        self.N0 = np.zeros(2 ** num_features, dtype=int)  # число "0" в кластерах
        self.total_N1 = np.sum(y)
        self.total_N0 = data_size - self.total_N1
        for k in range(data_size):
            code = self.bin_code(x[k, :])
            if y[k] == 1:
                self.N1[code] += 1
            else:
                self.N0[code] += 1

        """
        # рассчитываем AUC
        s = 0
        for code in range(2 ** num_features):
            s += max(total_N0 * N1[code], total_N1 * N0[code])
        if total_N0 == 0 or total_N1 == 0:
            auc = 1.0
        else:
            auc = s / (2 * total_N0 * total_N1)
        """

    def predict(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        y_pred = np.zeros(data_size, dtype=int)
        for k in range(data_size):
            code = self.bin_code(x[k, :])
            if self.N1[code] == 0 and self.N0[code] == 0:
                y_pred[k] = 0  #None
            elif self.N1[code] == 0:
                y_pred[k] = 0
            elif self.N0[code] == 0:
                y_pred[k] = 1
            else:  # N1[code] != 0 and N0[code] != 0
                if self.N1[code] / self.N0[code] >= self.total_N1 / self.total_N0:
                    y_pred[k] = 1
                # else:
                #   y_pred[k] = 0
        return y_pred


class BooleanFunctionMaxAUC:
    def __init__(self):
        self.cutoffs = None  # пороги для каждого предиктора
        self.clf = None
        self.num_features = None
        self.triples = None

    # находит оптимальные пороги
    def fit(self, x, y, use_genetic=True, set_cutoffs=None):
        data_size, num_features = x.shape[0], x.shape[1]
        self.num_features = num_features
        self.cutoffs = np.zeros(num_features)
        self.triples = None
        self.triples_logistic_model = None

        num_samples = 10000

        #np.random.seed(1234)
        lb = np.min(x, axis=0)
        ub = np.max(x, axis=0)

        m_d1 = 3
        n_e1 = 180
        lr = 0.1
        m_d = 2
        n_e = 100
        spw = 1

        def calc_J(c):
            z = np.zeros((data_size, num_features), dtype=int)
            for j in range(num_features):
                z[:, j] = np.where(x[:, j] >= c[j], 1, 0)
            #clf = BernoulliNB()
            #clf = RandomForestClassifier(n_estimators=n_e1, max_depth=m_d1)
            #clf = LogisticRegression()

            #clf = xgb.XGBClassifier(learning_rate=lr, eval_metric="auc", scale_pos_weight=spw,
            #                  max_depth=m_d, n_estimators=n_e)  # random_state?
            #clf.fit(z, y)

            #y_pred = clf.predict(z)
            #y_pred = clf.predict_proba(z)[:, 1]
            """
            # TODO: вынести в отдельный класс
            y_pred = np.zeros_like(y)

            def bin_code(x):
                #""
                #Найти двоичный код кластера
                #""
                ans = 0
                z = 1
                for i in range(num_features):
                    ans += z * x[i]
                    z *= 2
                return ans

            N1 = np.zeros(2 ** num_features, dtype=int)
            N0 = np.zeros(2 ** num_features, dtype=int)
            total_N1 = 0
            total_N0 = 0
            for k in range(data_size):
                code = bin_code(z[k, :])
                if y[k] == 1:
                    N1[code] += 1
                    total_N1 += 1
                else:
                    N0[code] += 1
                    total_N0 += 1
            # рассчитываем AUC
            s = 0
            for code in range(2 ** num_features):
                s += max(total_N0 * N1[code], total_N1 * N0[code])
            if total_N0 == 0 or total_N1 == 0:
                auc = 1.0
            else:
                auc = s / (2 * total_N0 * total_N1)
            """
            clf = BooleanClassifier()
            clf.fit(z, y)
            y_pred = clf.predict(z)

            fpr, tpr, _ = sklearn_metrics.roc_curve(y, y_pred)
            return sklearn_metrics.auc(fpr, tpr)
            #return auc

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


        varbound = np.array([[lb[j], ub[j]] for j in range(num_features)])
        #print(varbound)

        algorithm_param = {'max_num_iteration': 100,#1000, \
                           'population_size': 100, \
                           'mutation_probability': 0.1, \
                           'elit_ratio': 0.01, \
                           'crossover_probability': 0.5, \
                           'parents_portion': 0.3, \
                           'crossover_type': 'uniform', \
                           'max_iteration_without_improv': None}

        def f(c):
            return -calc_J(c)

        def fitness_func(ga_instance, solution, solution_idx):
            return calc_J(solution)

        num_generations = 100  # Number of generations.
        num_parents_mating = 10  # Number of solutions to be selected as parents in the mating pool.

        sol_per_pop = 20  # Number of solutions in the population.

        num_genes = num_features

        gene_space = [{'low': lb[j], 'high': ub[j]} if predictors[j] != "Killip class" else [1.6] for j in range(num_features)]

        def on_generation(ga_instance):
            pass
            #print(f"Generation = {ga_instance.generations_completed}")
            #print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")

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
           #print(f"Index of the best solution : {solution_idx}")




            #model = ga(function=f, dimension=num_features, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param, convergence_curve=True)#False)
            #model.run()
            #self.cutoffs = model.output_dict['variable']

            self.cutoffs = solution
        else:
            self.cutoffs = set_cutoffs

        z = np.zeros((data_size, num_features), dtype=int)
        for j in range(num_features):
            z[:, j] = np.where(x[:, j] >= self.cutoffs[j], 1, 0)

        clf = BooleanClassifier()
        clf.fit(z, y)
        self.clf = clf


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
        dontcares = []
        print('Переменные:', vars)
        for v in itertools.product([0, 1], repeat=self.num_features):
            #print(np.array(v).reshape(1, -1))
            #y_pred = self.clf.predict(np.array(v).reshape(1, -1))
            # TODO: сделать clf булевским классификатором!

            code = self.clf.bin_code(v)
            if self.clf.N1[code] == 0 and self.clf.N0[code] == 0:
                dontcares.append(v)
                #print(v, '->', '?')
            else:
                if self.clf.N0[code] == 0 or self.clf.N1[code] / self.clf.N0[code] >= self.clf.total_N1 / self.clf.total_N0:
                    #print(v, '->', 1)
                    minterms.append(v)
                else:
                    pass
                    #print(v, '->', 0)
        dnf = SOPform(vars, minterms, dontcares)
        print(dnf)  # вывод сокращенной ДНФ
        conjs = []  # список конъюнктов (список списков имен переменных)
        for mt in str(dnf).split("|"):
            v = list(map(lambda s: s.strip(), mt.split("&")))
            # убираем скобки с начала и с конца
            v[0] = v[0][1:]
            v[-1] = v[-1][:-1]
            #print(v)
            #assert(len(v) >= 5)
            if len(v) <= 5:
                conjs.append(v)
        #print(conjs)
        #print(predictors1)

        """
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

        
        num_triples = 30

        self.triples = []
        for t in triples:
            if t[1] >= triples[num_triples - 1][1]:
                self.triples.append(list(map(lambda a: predictors1.index(a), t[0])))
        """


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
#predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]

#predictors = ["HR", "Cr", "EF LV", "NEUT", "EOS", "Glu"]  #, "Killip class"]
predictors = ["Age", "Cr", "EF LV", "NEUT", "EOS"]

invert_predictors = find_predictors_to_invert(data, predictors)
data.prepare(predictors, "Dead", invert_predictors)

threshold = 0.025

num_splits = 1  #10
random_state = 123

np.random.seed(random_state)
random.seed(random_state)

csvfile = open('splits.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
csvwriter.writerow(["auc", "sen", "spec"])

csvwriter.writerow(["auc1", "sen1", "spec1", "auc2", "sen2", "spec2"])
for it in range(1, 1 + num_splits):
    print("SPLIT #", it, "of", num_splits)
    x_train_all, x_test_all, y_train_all, y_test_all = \
        train_test_split(data.x, data.y, test_size=0.2, stratify=data.y, random_state=random_state)  # закомментировать random_state


    skf = StratifiedKFold(n_splits=8)
    all_cutpoints = []
    #auc_history = []
    for fold, (train_index, test_index) in enumerate(skf.split(x_train_all, y_train_all)):
        x_train, x_test = data.x[train_index, :], data.x[test_index, :]
        y_train, y_test = data.y[train_index], data.y[test_index]
        print("  Fold", fold + 1)

        model1 = BooleanFunctionMaxAUC()
        model1.fit(x_train, y_train)
        print_model(model1, data)
        all_cutpoints.append(model1.cutoffs)

        test_model(model1, x_test, y_test, threshold)
        model1.interpret()

    # усредняем найденные пороги
    cutpoints = np.mean(np.vstack(all_cutpoints), axis=0)
    avg_model = BooleanFunctionMaxAUC()
    avg_model.fit(x_train_all, y_train_all, set_cutoffs=cutpoints)
    print_model(avg_model, data)
    auc1, sen1, spec1 = test_model(avg_model, x_test_all, y_test_all, threshold)
    avg_model.interpret()

    #print(model1.triples)
    """
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
    #auc2 = auc
    #sen2 = sensitivity
    #spec2 = specificity
    """

    #csvwriter.writerow(map(str, [auc1, sen1, spec1, auc2, sen2, spec2]))
