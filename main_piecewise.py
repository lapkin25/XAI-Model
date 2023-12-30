from sklearn.linear_model import LogisticRegression
from sklearn import metrics as sklearn_metrics
from read_data import read_data
from piecewise_model import PiecewiseModel

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
data_x, data_y = read_data(predictors, "Dead")

logist_reg = LogisticRegression()
logist_reg.fit(data_x, data_y)
y_pred = logist_reg.predict_proba(data_x)[:, 1]
auc = sklearn_metrics.roc_auc_score(data_y, y_pred)

print("Параметры логистической регрессии:")
print("Веса:", logist_reg.coef_.ravel())
print("Смещение:", logist_reg.intercept_[0])
print("AUC: ", auc)
print()

model = PiecewiseModel(logist_reg.coef_.ravel(), logist_reg.intercept_[0])
model.fit(data_x, data_y)

print("Параметры кусочной модели:")
print("Правые веса:", model.weights_plus)
print("Левые веса:", model.weights_minus)
print("Смещение:", model.intercept)
print("Пороги:", model.a)

y_prob = model.predict(data_x)
auc = sklearn_metrics.roc_auc_score(data_y, y_prob)
print("AUC: ", auc)
print()


