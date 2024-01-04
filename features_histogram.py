import math
import numpy as np
from read_data import read_data
import matplotlib.pyplot as plt
#import seaborn as sns

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
invert_predictors = [4, 6, 9]
data_x, data_y = read_data(predictors, "Dead")

# приводим все признаки к положительному влиянию на y
data_size, num_features = data_x.shape[0], data_x.shape[1]
for k in range(num_features):
    if k in invert_predictors:
        data_x[:, k] = 1 - data_x[:, k]

batch_size = 40

for k in range(num_features):
    ind = np.argsort(data_x[:, k])
    cnt = 0
    bars = []
    while cnt < data_size:
        cnt1 = 0
        cnt0 = 0
        for i in range(cnt, cnt + batch_size):
            if i >= data_size:
                break
            if data_y[ind[i]] == 1:
                cnt1 += 1
            else:
                cnt0 += 1
        x1 = data_x[ind[cnt], k]
        x2 = data_x[ind[min(cnt + batch_size - 1, data_size - 1)], k]
        cnt += batch_size
        P = (cnt1 + 0.5) / (cnt0 + cnt1 + 1)
        #P = cnt1 / (cnt0 + cnt1)
        bars.append({"P": P, "x1": x1, "x2": x2})
    #print(bars)
    #plt.bar(x=[(v["x1"] + v["x2"]) / 2 for v in bars], height=[v["P"] for v in bars], width=0.01)
    plt.bar(x=[v["x1"] for v in bars], height=[math.log(v["P"] / (1 - v["P"])) for v in bars], width=[(v["x2"] - v["x1"]) / 2 for v in bars], align='edge')
    plt.xlabel(predictors[k])
    plt.show()
    #sns.plt.bar([v["x1"] for v in bars], [v["P"] for v in bars], width=[(v["x2"] - v["x1"]) / 2 * 0.9 for v in bars])