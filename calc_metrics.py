import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from read_data import read_data

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
proba_threshold = 0.05

from tqdm import tqdm
for pivot in tqdm(range(data_size)):
    condition = np.all(data_x > data_x[pivot, :], axis=1)
    test_data_x = data_x[condition]
    test_data_y = data_y[condition]
    if len(data_y[condition]) == 0:
        continue
    y_pred = np.array(list(map(int, logist_reg.predict_proba(test_data_x)[:, 1] > proba_threshold)))
   # print("y_true =", test_data_y)
   # print("y_pred =", y_pred)
    if np.all(test_data_y == 1):
        continue
    #recall = metrics.recall_score(test_data_y, y_pred)
#    print(classification_report(test_data_y, y_pred, output_dict=True))
    tn, fp, fn, tp = metrics.confusion_matrix(test_data_y, y_pred).ravel()
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)
    if sens < 1:
        print("specificity =", spec, "sensitivity =", sens)
