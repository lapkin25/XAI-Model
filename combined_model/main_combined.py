import sys
sys.path.insert(1, '../dichotomization')
sys.path.insert(1, '../max_auc')

from dichotomization.read_data import Data
from max_auc.max_auc_model import InitialMaxAUCModel, IndividualMaxAUCModel
from combined_model import CombinedModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as sklearn_metrics


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


"""
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
"""


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

x_train, x_test, y_train, y_test = \
    train_test_split(data.x, data.y, test_size=0.2, stratify=data.y, random_state=123)  # закомментировать random_state

print("Начальная модель")
initial_model = InitialMaxAUCModel()
initial_model.fit(x_train, y_train)
auc0, sen0, spec0 = test_model(initial_model, x_test, y_test, threshold)

print("Непрерывная модель")
continuous_model = LogisticRegression()
continuous_model.fit(x_train, y_train)
auc1, sen1, spec1 = test_model(continuous_model, x_test, y_test, threshold)

print("Комбинированная модель")
combined_model = CombinedModel(initial_model, threshold=threshold)
combined_model.fit(x_train, y_train)
auc2, sen2, spec2 = test_model(combined_model, x_test, y_test, threshold)
