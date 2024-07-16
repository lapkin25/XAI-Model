import sys
sys.path.insert(1, '../dichotomization')
sys.path.insert(1, '../max_auc')

from dichotomization.read_data import Data
from max_auc.max_auc_model import InitialMaxAUCModel, IndividualMaxAUCModel
from max_auc.max_tpv_fpv_pairs import MinEntropyModel
from combined_model import CombinedModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as sklearn_metrics
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


def print_combined_model(model, data):
    print("=" * 10 + "\nМодель")
    print("Пороги:")
    for k, feature_name in enumerate(predictors):
        val = data.get_coord(feature_name, model.cutoffs[k])
        s = '≤' if feature_name in data.inverted_predictors else '≥'
        print(feature_name, " ", s, val, sep='')
    print("Веса:", model.weights)
    print("Интерсепт:", model.intercept)
    for k in range(len(predictors)):
        print("Вспомогательная модель для признака", predictors[k])
        for j in range(len(predictors)):
            feature_name = predictors[j]
            val = data.get_coord(feature_name, model.middle_cutoffs[k, j])
            s = '≤' if feature_name in data.inverted_predictors else '≥'
            print("  ", feature_name, " ", s, val, sep='')
        print("Веса:", model.middle_weights[k, :])
        print("Интерсепт:", model.middle_intercept[k])


def print_model(model, data):
    print("=" * 10 + "\nМодель")
    print("Пороги:")
    for k, feature_name in enumerate(predictors):
        val = data.get_coord(feature_name, model.cutoffs[k])
        s = '≤' if feature_name in data.inverted_predictors else '≥'
        print(feature_name, " ", s, val, sep='')
    print("Веса:", model.weights)
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


data = Data("DataSet.xlsx")
predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
invert_predictors = find_predictors_to_invert(data, predictors)
data.prepare(predictors, "Dead", invert_predictors)

threshold = 0.03

num_splits = 30

csvfile = open('splits.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
csvwriter.writerow(["auc0", "sen0", "spec0", "auc1", "sen1", "spec1", "auc2", "sen2", "spec2"])

for it in range(1, 1 + num_splits):
    print("SPLIT #", it, "of", num_splits)
    x_train, x_test, y_train, y_test = \
        train_test_split(data.x, data.y, test_size=0.2, stratify=data.y)  #, random_state=123)  # закомментировать random_state

    print("Начальная модель")
    initial_model = InitialMaxAUCModel()
    #initial_model = MinEntropyModel()
    initial_model.fit(x_train, y_train)
    print_model(initial_model, data)
    auc0, sen0, spec0 = test_model(initial_model, x_test, y_test, threshold)

    print("Непрерывная модель")
    continuous_model = LogisticRegression()
    continuous_model.fit(x_train, y_train)
    auc1, sen1, spec1 = test_model(continuous_model, x_test, y_test, threshold)

    print("Комбинированная модель")
    combined_model = CombinedModel(initial_model, threshold=threshold)
    #combined_model = CombinedModel(initial_model, threshold=threshold, method="entropy")
    combined_model.fit(x_train, y_train)
    print_combined_model(combined_model, data)
    auc2, sen2, spec2 = test_model(combined_model, x_test, y_test, threshold)

    csvwriter.writerow(map(str, [auc0, sen0, spec0, auc1, sen1, spec1, auc2, sen2, spec2]))
