import sys
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data
from max_auc.max_auc_model import InitialMaxAUCModel
import matplotlib.pyplot as plt
import numpy as np
import math


data = Data("DataSet.xlsx")
predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
data.prepare(predictors, "Dead", [], scale_data=False)  # не инвертируем предикторы

ind1 = 3 #8
ind2 = 8 #3
grid = np.linspace(np.min(data.x[:, ind1]), np.max(data.x[:, ind1]), 30, endpoint=False)
cutpoints = np.zeros_like(grid)
for i, cutoff in enumerate(grid):
    filtering_k = data.x[:, ind1] >= cutoff
    model = InitialMaxAUCModel()
    mid_x = np.c_[data.x[filtering_k, :ind1], data.x[filtering_k, ind1 + 1:]]
    mid_y = data.y[filtering_k]
    try:
        model.fit(mid_x, mid_y)
    except:
        break
    middle_cutoffs = np.concatenate((model.cutoffs[:ind1], [0.0], model.cutoffs[ind1:]))
    middle_weights = np.concatenate((model.weights[:ind1], [0.0], model.weights[ind1:]))
    middle_intercept = model.intercept
    print("Cutoff = ", middle_cutoffs[ind2])
    cutpoints[i] = middle_cutoffs[ind2]


plt.plot(grid, cutpoints)
plt.xlabel(predictors[ind1])
plt.ylabel(predictors[ind2])
plt.show()


