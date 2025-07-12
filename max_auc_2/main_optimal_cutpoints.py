import sys
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data
from sklearn.linear_model import LogisticRegression
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix, roc_auc_score, f1_score, accuracy_score, recall_score
from geneticalgorithm import geneticalgorithm as ga
import sklearn.metrics as sklearn_metrics
import pygad


class LogisticPairsModel:
    def __init__(self):
        self.logistic_model = None
        self._estimator_type = 'classifier'
        self.classes_ = np.array([0, 1])

    def get_pairs(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        num_pairs = num_features * (num_features - 1) // 2
        z = np.zeros((data_size, num_pairs), dtype=int)
        k = 0
        #print(x)
        for i in range(num_features):
            for j in range(i + 1, num_features):
                z[:, k] = x[:, i] & x[:, j]
                k += 1
        #print(z)
        return z

    def fit(self, x, y):
        # x - только бинарные признаки
        z = self.get_pairs(x)
        self.logistic_model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)
        self.logistic_model.fit(z, y)

    def predict_proba(self, x):
        # x - только бинарные признаки
        z = self.get_pairs(x)
        return self.logistic_model.predict_proba(z)

    #def predict(self, x):
        # x - только бинарные признаки
    #    return np.where(self.logistic_model.predict_proba(x) >= 0.5, 1, 0)

    def get_params(self, deep=False):
        return {}


class PhenotypesModel:
    def __init__(self):
        self.logistic_model = None
        self.phenotypes = None  # для каждого предиктора - список пар с этим предиктором, входящих в фенотип
        self._estimator_type = 'classifier'
        self.classes_ = np.array([0, 1])

    def get_phenotypes(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        z = np.zeros((data_size, num_features), dtype=int)
        for j in range(num_features):
            if len(self.phenotypes[j]) == 0:
                z[:, j] = x[:, j]
            else:
                for k in range(len(self.phenotypes[j])):
                    z[:, j] |= x[:, j] & x[:, self.phenotypes[j][k]]
        return z


    def fit(self, x, y):
        # x - только бинарные признаки
        data_size, num_features = x.shape[0], x.shape[1]
        self.phenotypes = [[] for _ in range(num_features)]
        for j in range(num_features):
            y_pred = x[:, j]
            auc_1 = sklearn_metrics.roc_auc_score(y, y_pred)  # AUC для отдельного предиктора
            for k in range(num_features):
                if j == k:
                    continue
                y_pred = x[:, j] & x[:, k]
                auc_2 = sklearn_metrics.roc_auc_score(y, y_pred)  # AUC для пары предикторов
                if auc_2 > auc_1:
                    self.phenotypes[j].append(k)

        z = self.get_phenotypes(x)

        self.logistic_model = LogisticRegression(max_iter=10000)
        self.logistic_model.fit(z, y)

    def predict_proba(self, x):
        # x - только бинарные признаки
        z = self.get_phenotypes(x)
        return self.logistic_model.predict_proba(z)

    def get_params(self, deep=False):
        return {}


class PairsModel:
    def __init__(self):
        self.cutpoints = None
        self.logistic_pairs_model = None

    def fit(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        self.cutpoints = np.zeros(num_features)

        lb = np.min(x, axis=0)
        ub = np.max(x, axis=0)
        #print(lb, ub)
        varbound = np.array([[lb[j], ub[j]] for j in range(num_features)])
        algorithm_param = {'max_num_iteration': 100,  # 1000,
                           'population_size': 100,
                           'mutation_probability': 0.1,
                           'elit_ratio': 0.01,
                           'crossover_probability': 0.5,
                           'parents_portion': 0.3,
                           'crossover_type': 'uniform',
                           'max_iteration_without_improv': None}

        def calc_J(c):
            z = np.zeros((data_size, num_features), dtype=int)
            for j in range(num_features):
                z[:, j] = np.where(x[:, j] >= c[j], 1, 0)

            def specificity_score(y_true, y_pred):
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                specificity = tn / (tn + fp)
                return specificity

            # Функция для преобразования вероятностей в бинарные метки с заданным порогом
            def custom_predict(proba, threshold=threshold):
                return (proba >= threshold).astype(int)

            # Пользовательская функция для расчета метрик с учетом порога
            def custom_metric(y_true, proba, metric_func, threshold=threshold):  # Устанавливаем порог 0.05
                y_pred = custom_predict(proba, threshold)
                return metric_func(y_true, y_pred)

            def custom_f1_score(y_true, proba, threshold=threshold):
                return custom_metric(y_true, proba, f1_score, threshold=threshold)

            def custom_accuracy_score(y_true, proba, threshold=threshold):
                return custom_metric(y_true, proba, accuracy_score, threshold=threshold)

            def custom_recall_score(y_true, proba, threshold=threshold):
                return custom_metric(y_true, proba, recall_score, threshold=threshold)

            def custom_specificity_score(y_true, proba, threshold=threshold):
                return custom_metric(y_true, proba, specificity_score, threshold=threshold)

            # Настроим метрики для кросс-валидации
            scoring = {'roc_auc': make_scorer(roc_auc_score, response_method='predict_proba'),
                       'f1': make_scorer(custom_f1_score, response_method='predict_proba', threshold=threshold),
                       'accuracy': make_scorer(custom_accuracy_score, response_method='predict_proba', threshold=threshold),
                       'sensitivity': make_scorer(custom_recall_score, response_method='predict_proba', threshold=threshold),
                       'specificity': make_scorer(custom_specificity_score, response_method='predict_proba', threshold=threshold)
                       }

            #logistic_pairs_model = LogisticPairsModel()
            logistic_pairs_model = PhenotypesModel()

            # Выполним кросс-валидацию с использованием cross_validate
            cv_results = cross_validate(logistic_pairs_model, z, y, cv=StratifiedKFold(n_splits=10),
                                        scoring=scoring, return_train_score=False)

            return np.mean(cv_results['test_roc_auc'])

        def f(c):
            val = calc_J(c)
            print(val)
            #return -calc_J(c)
            return -val

        """
        model = ga(function=f, dimension=num_features, variable_type='real', variable_boundaries=varbound,
                   algorithm_parameters=algorithm_param, convergence_curve=True)  # False)
        model.run()        
        self.cutpoints = model.output_dict['variable']
        """

        def calc_J_auc(c):
            z_train = np.zeros((x_train.shape[0], num_features), dtype=int)
            for j in range(num_features):
                z_train[:, j] = np.where(x_train[:, j] >= c[j], 1, 0)
            """
            z_test = np.zeros((x_test.shape[0], num_features), dtype=int)
            for j in range(num_features):
                z_test[:, j] = np.where(x_test[:, j] >= c[j], 1, 0)
            """

            #logistic_pairs_model = LogisticPairsModel()
            logistic_pairs_model = PhenotypesModel()

            logistic_pairs_model.fit(z_train, y_train)
            #y_pred = logistic_pairs_model.predict_proba(z_test)[:, 1]
            y_pred = logistic_pairs_model.predict_proba(z_train)[:, 1]
            auc_train = sklearn_metrics.roc_auc_score(y_train, y_pred)

            return auc_train

        def calc_J_loss(c):
            z_train = np.zeros((x_train.shape[0], num_features), dtype=int)
            for j in range(num_features):
                z_train[:, j] = np.where(x_train[:, j] >= c[j], 1, 0)
            logistic_pairs_model = PhenotypesModel()
            logistic_pairs_model.fit(z_train, y_train)
            y_pred = logistic_pairs_model.predict_proba(z_train)[:, 1]
            J_neg = np.sum(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred))
            return J_neg

        def fitness_func(ga_instance, solution, solution_idx):
            return calc_J_loss(solution)

        num_generations = 100  # Number of generations.
        num_parents_mating = 10  # Number of solutions to be selected as parents in the mating pool.

        sol_per_pop = 20  # Number of solutions in the population.
        num_genes = num_features

        gene_space = [{'low': lb[j], 'high': ub[j]} if predictors[j] != "Killip class" else [1.6] for j in range(num_features)]

        def on_generation(ga_instance):
            print(f"Generation = {ga_instance.generations_completed}")
            print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")

        """
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               fitness_func=fitness_func,
                               on_generation=on_generation,
                               gene_space=gene_space)


        # Running the GA to optimize the parameters of the function.
        ga_instance.run()

        #ga_instance.plot_fitness()

        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        print(f"Parameters of the best solution : {solution}")
        print(f"Fitness value of the best solution = {solution_fitness}")
        print(f"Index of the best solution : {solution_idx}")
        """

        skf = StratifiedKFold(n_splits=10)
        all_cutpoints = []
        auc_history = []
        for fold, (train_index, test_index) in enumerate(skf.split(x, y)):
            x_train, x_test = x[train_index, :], x[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            print("  Fold", fold + 1)

            ga_instance = pygad.GA(num_generations=num_generations,
                                   num_parents_mating=num_parents_mating,
                                   sol_per_pop=sol_per_pop,
                                   num_genes=num_genes,
                                   fitness_func=fitness_func,
                                   on_generation=on_generation,
                                   gene_space=gene_space)

            # Running the GA to optimize the parameters of the function.
            ga_instance.run()

            #ga_instance.plot_fitness()

            # Returning the details of the best solution.
            solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
            print(f"Parameters of the best solution : {solution}")
            print(f"Fitness value of the best solution = {solution_fitness}")
            print(f"Index of the best solution : {solution_idx}")

            all_cutpoints.append(solution)

            print("На тестовой части фолда")
            z_train = np.zeros((x_train.shape[0], num_features), dtype=int)
            for j in range(num_features):
                z_train[:, j] = np.where(x_train[:, j] >= solution[j], 1, 0)
            z_test = np.zeros((x_test.shape[0], num_features), dtype=int)
            for j in range(num_features):
                z_test[:, j] = np.where(x_test[:, j] >= solution[j], 1, 0)
            logistic_pairs_model = PhenotypesModel()
            logistic_pairs_model.fit(z_train, y_train)
            # y_pred = logistic_pairs_model.predict_proba(z_test)[:, 1]
            y_pred = logistic_pairs_model.predict_proba(z_test)[:, 1]
            auc_test = sklearn_metrics.roc_auc_score(y_test, y_pred)
            print("AUC =", auc_test)
            auc_history.append(auc_test)

        print("Кросс-валидация AUC", np.mean(auc_history))

        print(np.vstack(all_cutpoints))
        self.cutpoints = np.mean(np.vstack(all_cutpoints), axis=0)
        print(self.cutpoints)


        #solution = np.array([ 0.76530869,  0.0537224,   1.6,         1.86447204,  2.66886953,  0.67305026, 0.10364104,  7.55377103, -1.89968567,  1.91729093])

        # последний расчет (с помощью loss):
        # [0.79233634 0.74136715 1.6        1.66997645 2.44436646 0.79264738
        #  0.12555406 3.06368225 0.36896079 1.17035111]


        #skf = StratifiedKFold(n_splits=10)
        #for fold, (train_index, test_index) in enumerate(skf.split(x, y)):
        #    x_train, x_test = x[train_index, :], x[test_index, :]
        #    y_train, y_test = y[train_index], y[test_index]

        """
        for j in range(len(solution)):
            y_pred = (x[:, j] >= solution[j])
            auc_val = sklearn_metrics.roc_auc_score(y, y_pred)
            print("j =", j, "  auc =", auc_val)
            for k in range(j + 1, len(solution)):
                y_pred = (x[:, j] >= solution[j]) & (x[:, k] >= solution[k])
                auc_val = sklearn_metrics.roc_auc_score(y, y_pred)
                print("   j =", j, "& k =", k, " auc =", auc_val)

        y_pred = (x[:, 0] >= solution[0]) & ((x[:, 5] >= solution[5]) | (x[:, 6] >= solution[6]))
        auc_val = sklearn_metrics.roc_auc_score(y, y_pred)
        print("фенотип ", auc_val)
        """

        # покоординатное улучшение...
        """
        best_J = calc_J(solution)
        print(solution, best_J)
        for j in range(len(solution)):
            print("j =", j)
            new_solution = solution.copy()
            for val in np.linspace(lb[j], ub[j]):
                new_solution[j] = val
                J = calc_J(new_solution)
                if J > best_J:
                    print('test')
                    solution = new_solution.copy()
                    best_J = J
            print(solution, best_J)
        """

        #self.cutpoints = solution


        # далее обучить логистическую регрессию на парах
        z = np.zeros((data_size, num_features), dtype=int)
        for j in range(num_features):
            z[:, j] = np.where(x[:, j] >= self.cutpoints[j], 1, 0)
        #self.logistic_pairs_model = LogisticPairsModel()
        self.logistic_pairs_model = PhenotypesModel()
        self.logistic_pairs_model.fit(z, y)

    def predict_proba(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        z = np.zeros((data_size, num_features), dtype=int)
        for j in range(num_features):
            z[:, j] = np.where(x[:, j] >= self.cutpoints[j], 1, 0)
        return self.logistic_pairs_model.predict_proba(z)


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


def print_model(model, data, phenotypes=True):
    print("=" * 10 + "\nМодель")
    print("Пороги:")
    for k, feature_name in enumerate(predictors):
        val = data.get_coord(feature_name, model.cutpoints[k])
        s = '≤' if feature_name in data.inverted_predictors else '≥'
        print(feature_name, " ", s, val, sep='')
    if phenotypes:
        print("Фенотипы:")
        for j in range(len(predictors)):
            if len(model.logistic_pairs_model.phenotypes[j]) == 0:
                print(predictors[j])
            else:
                print(predictors[j], '& (', end='')
                for k in range(len(model.logistic_pairs_model.phenotypes[j])):
                    if k > 0:
                        print(' | ', end='')
                    print(predictors[model.logistic_pairs_model.phenotypes[j][k]], end='')
                print(')')
        print("Веса:", model.logistic_pairs_model.logistic_model.coef_.ravel())
    else:
        print("Пары:")
        k = 0
        # print(x)
        num_features = len(predictors)
        for i in range(num_features):
            for j in range(i + 1, num_features):
                print(predictors[i], predictors[j], model.logistic_pairs_model.logistic_model.coef_.ravel()[k])
                k += 1


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
csvwriter.writerow(["auc_cv", "sen_cv", "spec_cv", "auc_test", "sen_test", "spec_test"])

for it in range(1, 1 + num_splits):
    np.random.seed(random_state + it - 1)
    print("SPLIT #", it, "of", num_splits)
    x_train, x_test, y_train, y_test = \
        train_test_split(data.x, data.y, test_size=0.2, stratify=data.y, random_state=random_state)  # закомментировать random_state

    model = PairsModel()
    #auc_cv, sens_cv, spec_cv = model.fit(x_train, y_train)
    model.fit(x_train, y_train)
    print_model(model, data)
    auc_test, sen_test, spec_test = test_model(model, x_test, y_test, threshold)

