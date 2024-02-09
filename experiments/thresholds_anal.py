from read_data import read_data
import itertools

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
invert_predictors = [4, 6, 9]
data_x, data_y = read_data(predictors, "Dead")

selected_predictors = ["HR", "EF LV", "EOS", "Glu", "Killip class"]
num_selected_predictors = len(selected_predictors)
thresholds = {"HR": 0.570, "EF LV": 0.633, "EOS": 0.992, "Glu": 0.289, "Killip class": 0.867}

# приводим все признаки к положительному влиянию на y
data_size, num_features = data_x.shape[0], data_x.shape[1]
for k in range(num_features):
    if k in invert_predictors:
        data_x[:, k] = 1 - data_x[:, k]

cnt1 = sum(data_y)
P = cnt1 / data_size
print("Apriori P = ", P)

for comb in itertools.combinations(selected_predictors, 3):
    print(comb)
    cnt1 = 0
    cnt0 = 0
    for i in range(data_size):
        forall = True
        for predictor in comb:
            j = predictors.index(predictor)
            forall = forall and data_x[i][j] > thresholds[predictor]
        if forall:
            if data_y[i] == 1:
                cnt1 += 1
            else:
                cnt0 += 1
    if (cnt1 + cnt0 == 0):
        P = None
    else:
        P = cnt1 / (cnt1 + cnt0)
    print(P * 100, "%  of ", cnt1 + cnt0)

