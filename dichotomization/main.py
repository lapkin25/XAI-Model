from read_data import Data
from sklearn.linear_model import LogisticRegression
from initial_model import InitialModel
import numpy as np
from sklearn import metrics as sklearn_metrics
from sklearn.model_selection import StratifiedKFold, train_test_split


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
"""
initial_model.fit(data.x, data.y)
print("Начальная модель")
#print("Пороги:", initial_model.cutoffs)
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
"""

print("Кросс-валидация")
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=123)
for i, (train_index, test_index) in enumerate(skf.split(data.x, data.y)):
    print(f"Fold {i}:")
    x_train = data.x[train_index, :]
    y_train = data.y[train_index]
    x_test = data.x[test_index, :]
    y_test = data.y[test_index]
    initial_model.fit(x_train, y_train)
    p = initial_model.predict_proba(x_test, y_test)
    auc = sklearn_metrics.roc_auc_score(y_test, p)
    print("AUC:", auc)

print("Итоговое тестирование")
x_train, x_test, y_train, y_test =\
    train_test_split(data.x, data.y, test_size=0.2, random_state=123, stratify=data.y)
initial_model.fit(x_train, y_train)
p = initial_model.predict_proba(x_test, y_test)
auc = sklearn_metrics.roc_auc_score(y_test, p)
print("AUC:", auc)
