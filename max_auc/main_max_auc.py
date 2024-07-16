import sys
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data
from max_auc_model import InitialMaxAUCModel, IndividualMaxAUCModel,\
    CombinedMaxAUCModel, SelectedCombinedMaxAUCModel, RandomForest
from max_tpv_fpv_pairs import AllPairs
from extract_rules import ExtractRules
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as sklearn_metrics
from permetrics.classification import ClassificationMetric
import matplotlib.pyplot as plt
import csv


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
        y_pred = np.zeros(data.x.shape[0])
        for i in range(data.x.shape[0]):
            if data.x[i, k] >= model.cutoffs[k] and data.x[i, j] >= xj_cutoff:
                y_pred[i] = 1
                if data.y[i] == 1:
                    tp += 1
                else:
                    fp += 1
        print("   TP = ", tp, " FP = ", fp, " Prob = ", tp / (tp + fp))
        cm = ClassificationMetric(data.y, y_pred)
        print("   Gini = ", cm.gini_index(average=None))
    print("Веса:", model.individual_weights)
    print("Комбинированные веса:", model.combined_weights)
    print("Интерсепт:", model.intercept)


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


def test_rules_count(model, x_test, y_test):
    rules_cnt = model.count_rules(x_test)
    rules_cnt_ones = []
    rules_cnt_zeros = []
    for i in range(rules_cnt.shape[0]):
        if y_test[i] == 1:
            rules_cnt_ones.append(rules_cnt[i])
        else:
            rules_cnt_zeros.append(rules_cnt[i])

    print("В среднем на 1:", np.mean(rules_cnt_ones))
    print("В среднем на 0:", np.mean(rules_cnt_zeros))

    xx = list(range(min(rules_cnt_ones), max(rules_cnt_ones) + 1))
    yy = [rules_cnt_ones.count(a) for a in xx]
    b = plt.bar(xx, yy)
    plt.bar_label(b, labels=yy)
    plt.show()

    xx = list(range(min(rules_cnt_zeros), max(rules_cnt_zeros) + 1))
    yy = [rules_cnt_zeros.count(a) for a in xx]
    b = plt.bar(xx, yy)
    plt.bar_label(b, labels=yy)
    plt.show()


data = Data("DataSet.xlsx")
predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
invert_predictors = find_predictors_to_invert(data, predictors)
data.prepare(predictors, "Dead", invert_predictors)

threshold = 0.03  #0.04
num_combined_features = 12  #10

num_splits = 30

csvfile = open('splits.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
csvwriter.writerow(["auc0", "sen0", "spec0", "auc1", "sen1", "spec1", "auc2", "sen2", "spec2", "auc3", "sen3", "spec3"])
# раскомментировать - для метода минимальной энтропии
# csvwriter.writerow(["auc1", "sen1", "spec1", "auc2", "sen2", "spec2"])
for it in range(1, 1 + num_splits):
    print("SPLIT #", it, "of", num_splits)
    x_train, x_test, y_train, y_test = \
        train_test_split(data.x, data.y, test_size=0.2, stratify=data.y)  #, random_state=123)  # закомментировать random_state

    initial_model = InitialMaxAUCModel()
    initial_model.fit(x_train, y_train)
    print("Начальная модель")
    #print_model(initial_model, data)
    auc0, sen0, spec0 = test_model(initial_model, x_test, y_test, threshold)

    ind_model = IndividualMaxAUCModel(verbose_training=True,
                                      training_iterations=6)
    ind_model.fit(x_train, y_train)
    print("Модель с индивидуальными признаками")
    #print_model(ind_model, data)
    auc2, sen2, spec2 = test_model(ind_model, x_test, y_test, threshold)

    """ раскомментировать - для метода минимальной энтропии
    # TODO: вынести индивидуальные пороги в отдельную модель
    initial_model = InitialMaxAUCModel()
    initial_model.fit(x_train, y_train)
    all_pairs = AllPairs(initial_model)
    #all_pairs = AllPairs(ind_model)
    ##all_pairs.fit(x_train, y_train)
    #all_pairs.fit_auc(x_train, y_train)
    all_pairs.fit_entropy(x_train, y_train)
    #print_model(all_pairs, data)

    extract_rules = ExtractRules(all_pairs, 35)
    extract_rules.fit(x_train, y_train)
    print_model(extract_rules, data)
    auc3, sen3, spec3 = test_model(extract_rules, x_test, y_test, threshold)

    #test_model(extract_rules, data.x, data.y, threshold)

    test_rules_count(extract_rules, data.x, data.y)

    all_pairs1 = AllPairs(initial_model)
    all_pairs1.fit_entropy(x_train, y_train, simplified=True)
    extract_rules1 = ExtractRules(all_pairs1, 35)
    extract_rules1.fit(x_train, y_train)
    print_model(extract_rules1, data)
    auc4, sen4, spec4 = test_model(extract_rules1, x_test, y_test, threshold)
    """

    """
    rf = RandomForest(all_pairs, K=30)
    rf.fit(x_train, y_train)
    print("Модель на основе случайного леса")
    print_model(rf, data)
    auc3, sen3, spec3 = test_model(rf, x_test, y_test, threshold)
    """

    """
    sel_model = SelectedCombinedMaxAUCModel(all_pairs, verbose_training=True, K=num_combined_features)
    sel_model.fit(x_train, y_train)
    print("Модель с выбранными комбинированными признаками")
    print_model(sel_model, data)
    auc3, sen3, spec3 = test_model(sel_model, x_test, y_test, threshold)
    """

    # непрерывная модель
    print("Непрерывная модель")
    continuous_model = LogisticRegression()
    continuous_model.fit(x_train, y_train)
    auc1, sen1, spec1 = test_model(continuous_model, x_test, y_test, threshold)

    model = CombinedMaxAUCModel(ind_model, verbose_training=True, K=num_combined_features,
                                combined_training_iterations=5, refresh_features=True)
    model.fit(x_train, y_train)
    print("Модель с комбинированными признаками")
    print_model(model, data)
    auc3, sen3, spec3 = test_model(model, x_test, y_test, threshold)

    csvwriter.writerow(map(str, [auc0, sen0, spec0, auc1, sen1, spec1, auc2, sen2, spec2, auc3, sen3, spec3]))
    # раскомментировать - для метода минимальной энтропии
    # csvwriter.writerow(map(str, [auc3, sen3, spec3, auc4, sen4, spec4]))

"""
    # проверяем модель
    print("Проверка модели 2")
    #ind_model = IndividualMaxAUCModel(verbose_training=False,
    #    training_iterations=20)
    #ind_model.fit(x_train, y_train)
    #test_model(ind_model, x_test, y_test, threshold)
    bin_x_train = ind_model.dichotomize(x_train)
    bin_x_test = ind_model.dichotomize(x_test)
    model2 = LogisticRegression()
    model2.fit(bin_x_train, y_train)
    #print("Веса =", model2.coef_)
    auc2, sen2, spec2 = test_model(model2, bin_x_test, y_test, threshold)

    # проверяем модель
    print("Проверка модели 3")
    bin_x_train_combined = model.dichotomize_combined(x_train)
    bin_x_test_combined = model.dichotomize_combined(x_test)
    model3 = LogisticRegression()
    model3.fit(bin_x_train_combined, y_train)
    #print("Веса =", model3.coef_)
    auc3, sen3, spec3 = test_model(model3, bin_x_test_combined, y_test, threshold)

    csvwriter.writerow(map(str, [auc1, sen1, spec1, auc2, sen2, spec2, auc3, sen3, spec3]))
"""