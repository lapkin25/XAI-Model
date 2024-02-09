import math
import numpy as np
from read_data import read_data

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
invert_predictors = [4, 6, 9]
data_x, data_y = read_data(predictors, "Dead")

# приводим все признаки к положительному влиянию на y
data_size, num_features = data_x.shape[0], data_x.shape[1]
for k in range(num_features):
    if k in invert_predictors:
        data_x[:, k] = 1 - data_x[:, k]

points_num = np.array([0] * data_size)
points_1_num = np.array([0] * data_size)
from tqdm import tqdm
for pivot in tqdm(range(data_size)):
    cnt = 0
    cnt1 = 0
    for i in range(data_size):
        if all([data_x[i][j] >= data_x[pivot][j] for j in range(num_features)]):
            cnt += 1
            if data_y[i] == 1:
                cnt1 += 1
    points_num[pivot] = cnt
    points_1_num[pivot] = cnt1

ind = np.argsort(points_num)
ans_ind = []
for i in range(data_size):
    P = points_1_num[ind[i]] / points_num[ind[i]]
    if P > 0.3:
        ans_ind.append(ind[i])
        #print(points_num[ind[i]], points_1_num[ind[i]], P)
        #print(data_x[ind[i]])

for i in range(len(ans_ind)):
    is_ans = True
    for k in range(len(ans_ind)):
        if k != i and all([data_x[ans_ind[i]][j] >= data_x[ans_ind[k]][j] for j in range(num_features)]):
            is_ans = False
    if is_ans:
        P = points_1_num[ans_ind[i]] / points_num[ans_ind[i]]
        print(points_num[ans_ind[i]], points_1_num[ans_ind[i]], P)
        print(data_x[ans_ind[i]])
        print()


