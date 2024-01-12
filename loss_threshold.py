import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from read_data import read_data
import matplotlib.pyplot as plt


def calc_loss(x, y):
    logist_reg = LogisticRegression()  # (penalty=None)
    logist_reg.fit(x, y)
    y_pred = logist_reg.predict_proba(x)
    return log_loss(y, y_pred), log_loss(y, y_pred) * len(y)


def calc_f2(x, y):
    logist_reg = LogisticRegression()
    logist_reg.fit(x, y)
    y_pred = logist_reg.predict(x)
    return f1_score(y, y_pred)


def calc_intercept(x, y):
    # TODO: обучить модель с фиксированными коэффициентами и найти свободный член
    # def calc_intercept(x, y, coef):
    logist_reg = LogisticRegression()
    logist_reg.fit(x, y)
    return logist_reg.intercept_[0]



predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
invert_predictors = [4, 6, 9]
data_x, data_y = read_data(predictors, "Dead")

# приводим все признаки к положительному влиянию на y
data_size, num_features = data_x.shape[0], data_x.shape[1]
for k in range(num_features):
    if k in invert_predictors:
        data_x[:, k] = 1 - data_x[:, k]

#feature = 8
"""
for feature in range(10):
    plot_a = []
    plot_loss1 = []
    plot_loss2 = []
    print(predictors[feature])
    for a in np.linspace(0.0, 1.0, num=30, endpoint=False):
        if all(data_y[data_x[:, feature] > a]) or all(1 - data_y[data_x[:, feature] > a]):
            continue
        loss1, loss2 = calc_loss(data_x[data_x[:, feature] > a, feature].reshape(-1, 1), data_y[data_x[:, feature] > a])
        print(a, loss1, loss2)
        plot_a.append(a)
        plot_loss1.append(loss1)
        plot_loss2.append(loss2)
    plt.plot(plot_a, plot_loss1)
    plt.xlabel(predictors[feature] + " - порог a")
    plt.ylabel("loss (x > a)")
    plt.show()
    plt.plot(plot_a, plot_loss2)
    plt.xlabel(predictors[feature] + " - порог a")
    plt.ylabel("N * loss (x > a)")
    plt.show()
"""

feature1 = 8
feature2 = 9
K = 30
X1 = np.linspace(0.0, 1.0, num=K, endpoint=False)
Y1 = np.linspace(0.0, 1.0, num=K, endpoint=False)
X, Y = np.meshgrid(X1, Y1)
Z1 = np.zeros(shape=(K, K))
Z2 = np.zeros(shape=(K, K))
for i in range(K):
    for j in range(K):
        a1 = X1[i]
        a2 = Y1[j]
        selected = (data_x[:, feature1] > a1) & (data_x[:, feature2] > a2)
        if all(data_y[selected]) or all(1 - data_y[selected]):
            continue
        """
        loss1, loss2 = calc_loss(
            np.hstack((data_x[selected, feature1].reshape(-1, 1),
                      data_x[selected, feature2].reshape(-1, 1))),
            data_y[selected])
        """

        #loss1, loss2 = calc_loss(data_x[selected, :], data_y[selected])

        #loss1 = calc_f2(data_x[selected, :], data_y[selected])
        #loss2 = loss1

        loss1 = calc_intercept(data_x[selected, :], data_y[selected])
        loss2 = loss1

        Z1[i, j] = loss1
        Z2[i, j] = loss2
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z2)
ax.set_xlabel(predictors[feature1] + " - порог a")
ax.set_ylabel(predictors[feature2] + " - порог a")
ax.set_zlabel("N * loss (x1 > a1 & x2 > a2)")
plt.show()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z1)
ax.set_xlabel(predictors[feature1] + " - порог a")
ax.set_ylabel(predictors[feature2] + " - порог a")
ax.set_zlabel("loss (x1 > a1 & x2 > a2)")
plt.show()