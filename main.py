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

#print(math.log(5 / 95))

print("Параметры логистической регрессии:")
print("Веса:", logist_reg.coef_.ravel())
print("Смещение:", logist_reg.intercept_[0])
print("AUC: ", auc)
print()

K = 25
emp_log_odds, neighbors1 = empirical_log_odds(data_x, data_y, K, logist_reg.coef_.ravel())
pred_log_odds = list(map(lambda z: math.log(z / (1 - z)), y_pred))
for i in range(data_size):
    print(emp_log_odds[i], pred_log_odds[i], neighbors1[i], '  ', data_y[i], y_pred[i])
plt.scatter(pred_log_odds, emp_log_odds, color=list(map(lambda y: 'r' if y == 1 else 'g', data_y)))
plt.plot([min(emp_log_odds), max(emp_log_odds)], [min(emp_log_odds), max(emp_log_odds)], 'b')
plt.plot([min(pred_log_odds), max(pred_log_odds)], [-2.94, -2.94], 'b')
plt.plot([-2.94, -2.94], [min(emp_log_odds), max(emp_log_odds)], 'b')
plt.xlabel("Предсказанный логит")
plt.ylabel("Эмпирический логит")
plt.show()

diff_pred_emp = []
selected_data = []
for i in range(data_size):
    if data_y[i] == 1:
        diff_pred_emp.append(emp_log_odds[i] - pred_log_odds[i])
        selected_data.append(data_x[i])

selected_data_size = len(selected_data)

for j in range(num_features):
    data_j = [selected_data[i][j] for i in range(selected_data_size)]
    plt.scatter(data_j, diff_pred_emp)
    plt.plot([min(data_j), max(data_j)], [0, 0], 'k')
    plt.xlabel(predictors[j])
    plt.ylabel("emp_logit - pred_logit")
    plt.savefig('results/' + predictors[j] + '.png')
    plt.show()

