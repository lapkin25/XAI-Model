import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from fixed_model import FixedModel
from sklearn import metrics
from read_data import read_data
import matplotlib.pyplot as plt

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
invert_predictors = [4, 6, 9]
data_x, data_y = read_data(predictors, "Dead")

# приводим все признаки к положительному влиянию на y
data_size, num_features = data_x.shape[0], data_x.shape[1]
for k in range(num_features):
    if k in invert_predictors:
        data_x[:, k] = 1 - data_x[:, k]

logist_reg = LogisticRegression()
logist_reg.fit(data_x, data_y)
y_pred = logist_reg.predict_proba(data_x)[:, 1]
auc = metrics.roc_auc_score(data_y, y_pred)

print("Параметры логистической регрессии:")
print("Веса:", logist_reg.coef_.ravel())
print("Смещение:", logist_reg.intercept_[0])
print("AUC: ", auc)
print()

"""
for j in range(len(predictors)):
    plt.scatter(data_x[data_y == 0, j], np.log(y_pred / (1 - y_pred))[data_y == 0], color='b')
    plt.scatter(data_x[data_y == 1, j], np.log(y_pred / (1 - y_pred))[data_y == 1], color='r')
    plt.xlabel(predictors[j])
    plt.ylabel("predicted logit")
    plt.show()
"""

fixed_model = FixedModel(logist_reg.coef_.ravel(), logist_reg.intercept_[0])
fixed_model.fit(data_x, data_y)
print("Параметры модели с фиксированными весами:")
print("Веса:", fixed_model.weights)
print("Смещение:", fixed_model.intercept)


feature = 3  # номер признака
min_intercept = -12.5  # значения ниже этого нас не интересуют
x_plot = []
y_plot = []
for threshold in np.linspace(0, 0.9, 30):
    condition = (data_x[:, feature] > threshold) # & (data_x[:, 0] > 0.5)
    test_data_x = data_x[condition]
    test_data_y = data_y[condition]
    if len(data_y[condition]) == 0:
        continue
    fixed_model.fit(test_data_x, test_data_y)
    if fixed_model.intercept > min_intercept:
        print("Порог:", threshold)
        print("Точек", np.sum(condition))
        print("w0 =", fixed_model.intercept)
        print()
    x_plot.append(threshold)
    y_plot.append(fixed_model.intercept)
plt.plot(x_plot, y_plot)
plt.xlabel("Порог (выше которого оставляем точки)")
plt.ylabel("Интерсепт")
plt.show()



# TODO: перебрать пары и тройки признаков; вывести таблицу порогов при w0 > -12.5

feature1 = 4  # номер признака
feature2 = 3
min_intercept = -12.5  # значения ниже этого нас не интересуют
min_points = 50
x_plot = []
y_plot = []
for threshold1 in np.linspace(0, 0.9, 30):
    for threshold2 in np.linspace(0, 0.9, 30):
        condition = (data_x[:, feature1] > threshold1) & (data_x[:, feature2] > threshold2)
        test_data_x = data_x[condition]
        test_data_y = data_y[condition]
        if len(data_y[condition]) == 0:
            continue
        fixed_model.fit(test_data_x, test_data_y)
        if fixed_model.intercept > min_intercept and np.sum(condition) >= min_points:
            print("Порог 1:", threshold1)
            print("Порог 2:", threshold2)
            print("Точек", np.sum(condition))
            print("w0 =", fixed_model.intercept)
            print()



"""
from tqdm import tqdm
for pivot in tqdm(range(data_size)):
    condition = np.all(data_x > data_x[pivot, :], axis=1)
    test_data_x = data_x[condition]
    test_data_y = data_y[condition]
    if len(data_y[condition]) == 0:
        continue
    fixed_model.fit(test_data_x, test_data_y)
    if fixed_model.intercept > logist_reg.intercept_[0]:
        print(data_x[pivot])
        print("Точек", np.sum(condition))
        print("w0 =", fixed_model.intercept)
        print()
"""