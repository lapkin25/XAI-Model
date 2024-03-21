import sys
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data
from new_combined_features_model import NewCombinedFeaturesModel, NewIndividualFeaturesModel
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


data = Data("DataSet.xlsx")
predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
invert_predictors = find_predictors_to_invert(data, predictors)
data.prepare(predictors, "Dead", invert_predictors)

threshold = 0.04
num_combined_features = 30

num_splits = 30
for it in range(1, 1 + num_splits):
    print("SPLIT #", it, "of", num_splits)
    x_train, x_test, y_train, y_test = \
        train_test_split(data.x, data.y, test_size=0.2, stratify=data.y)  #, random_state=123)

    model = NewCombinedFeaturesModel(verbose_training=False, p0=threshold,
        K=num_combined_features, delta_a=0.2, delta_w=0.3,
        individual_training_iterations=25, combined_training_iterations=10)
    model.fit(x_train, y_train)
    model.fit_intercept(x_train, y_train)

    print_model(model, data)
    test_model(model, x_test, y_test, threshold)

    # проверяем модель
    print("Проверка модели 1")
    model1 = LogisticRegression()
    model1.fit(x_train, y_train)
    test_model(model1, x_test, y_test, threshold)

    # проверяем модель
    print("Проверка модели 2")
    ind_model = NewIndividualFeaturesModel(verbose_training=False, p0=threshold,
        delta_a=0.2, delta_w=0.3, training_iterations=25)
    ind_model.fit(x_train, y_train)
    ind_model.fit_intercept(x_train, y_train)
    #test_model(ind_model, x_test, y_test, threshold)
    bin_x_train = ind_model.dichotomize(x_train)
    bin_x_test = ind_model.dichotomize(x_test)
    model2 = LogisticRegression()
    model2.fit(bin_x_train, y_train)
    #print("Веса =", model2.coef_)
    test_model(model2, bin_x_test, y_test, threshold)

    # проверяем модель
    print("Проверка модели 3")
    bin_x_train_combined = model.dichotomize_combined(x_train)
    bin_x_test_combined = model.dichotomize_combined(x_test)
    model3 = LogisticRegression()
    model3.fit(bin_x_train_combined, y_train)
    #print("Веса =", model3.coef_)
    test_model(model3, bin_x_test_combined, y_test, threshold)
