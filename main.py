import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as sklearn_metrics
from read_data import read_data
from empirical_odds import empirical_log_odds

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
data_x, data_y = read_data(predictors, "Dead")
data_size, num_features = data_x.shape[0], data_x.shape[1]

logist_reg = LogisticRegression()
logist_reg.fit(data_x, data_y)
y_pred = logist_reg.predict_proba(data_x)[:, 1]
auc = sklearn_metrics.roc_auc_score(data_y, y_pred)

print(math.log(5 / 95))

print("Параметры логистической регрессии:")
print("Веса:", logist_reg.coef_.ravel())
print("Смещение:", logist_reg.intercept_[0])
print("AUC: ", auc)
print()

K = 20
emp_log_odds = empirical_log_odds(data_x, data_y, K, logist_reg.coef_.ravel())
pred_log_odds = list(map(lambda z: math.log(z / (1 - z)), y_pred))
for i in range(data_size):
    print(emp_log_odds[i], pred_log_odds[i])
plt.scatter(pred_log_odds, emp_log_odds, color=list(map(lambda y: 'r' if y == 1 else 'g', data_y)))
plt.plot([min(emp_log_odds), max(emp_log_odds)], [min(emp_log_odds), max(emp_log_odds)], 'b')
plt.plot([min(emp_log_odds), max(emp_log_odds)], [-2.94, -2.94], 'g')
plt.xlabel("Предсказанный логит")
plt.ylabel("Эмпирический логит")
plt.show()

