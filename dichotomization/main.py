from read_data import Data
from sklearn.linear_model import LogisticRegression
from initial_model import InitialModel
import numpy as np
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


data = Data("DataSet.xlsx")
predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
invert_predictors = find_predictors_to_invert(data, predictors)
print("Признаки с отрицательными весами:", invert_predictors)
data.prepare(predictors, "Dead", invert_predictors)

threshold = 0.05

initial_model = InitialModel()
# TODO: кросс-валидация и итоговое тестирование
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