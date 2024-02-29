import numpy as np
from read_data import Data
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as sklearn_metrics
from initial_model import InitialModel
from adjusted_model import AdjustedModel
from combined_features_model import CombinedFeaturesModel
from model_testing import test_model


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


data = Data("DataSet.xlsx")
predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
invert_predictors = find_predictors_to_invert(data, predictors)
print("Признаки с отрицательными весами:", invert_predictors)
data.prepare(predictors, "Dead", invert_predictors)

threshold = 0.05

initial_model = InitialModel()

initial_model.fit(data.x, data.y)
print("Начальная модель")
print("Пороги:")
for k, feature_name in enumerate(predictors):
    val = data.get_coord(feature_name, initial_model.cutoffs[k])
    s = '<' if feature_name in invert_predictors else '>'
    print(feature_name, " ", s, val, sep='')

print("Веса:", initial_model.weights)
print("Интерсепт:", initial_model.intercept)
p = initial_model.predict_proba(data.x, data.y)
y_pred = np.where(p >= 0.05, 1, 0)
auc = sklearn_metrics.roc_auc_score(data.y, p)
print("AUC:", auc)

test_model(initial_model, data)

adjusted_model = CombinedFeaturesModel()  #AdjustedModel()
adjusted_model.fit(data.x, data.y)
print("=" * 10 + "\nНастроенная модель")
print("Пороги:")
for k, feature_name in enumerate(predictors):
    val = data.get_coord(feature_name, adjusted_model.cutoffs[k])
    s = '≤' if feature_name in invert_predictors else '≥'
    print(feature_name, " ", s, val, sep='')
print("Комбинированные пороги:")
for k, j, xj_cutoff in adjusted_model.combined_features:
    feature_name = predictors[k]
    val = data.get_coord(feature_name, adjusted_model.cutoffs[k])
    s = '≤' if feature_name in invert_predictors else '≥'
    print(feature_name, " ", s, val, sep='', end='')
    feature_name = predictors[j]
    val = data.get_coord(feature_name, xj_cutoff)
    s = '≤' if feature_name in invert_predictors else '≥'
    print(" & ", feature_name, " ", s, val, sep='')

print("Веса:", adjusted_model.weights)
print("Комбинированные веса:", adjusted_model.combined_weights)
print("Интерсепт:", adjusted_model.intercept)
p = adjusted_model.predict_proba(data.x, data.y)
y_pred = np.where(p >= 0.05, 1, 0)
# TODO: вывести матрицу неточностей
auc = sklearn_metrics.roc_auc_score(data.y, p)
print("AUC:", auc)

test_model(adjusted_model, data)

adjusted_model.fit_logistic(data.x, data.y)
print("После обучения логистической регрессии при заданных порогах:")
print("Веса:", adjusted_model.weights)
print("Интерсепт:", adjusted_model.intercept)
p = adjusted_model.predict_proba(data.x, data.y)
y_pred = np.where(p >= 0.05, 1, 0)
# TODO: вывести матрицу неточностей
auc = sklearn_metrics.roc_auc_score(data.y, p)
print("AUC:", auc)


# TODO: унаследовать AdjustedModel и InitialModel от класса Model,
#   вынести повторяющийся код в базовый класс
