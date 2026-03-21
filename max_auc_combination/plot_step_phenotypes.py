import sys

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data
from sklearn.linear_model import LogisticRegression
import csv
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix
import sklearn.metrics as sklearn_metrics
import math
#from pyDOE import lhs
#from geneticalgorithm import geneticalgorithm as ga
#from permetrics.classification import ClassificationMetric
from sklearn.naive_bayes import BernoulliNB
import itertools
from sympy.logic import SOPform
from sympy import symbols
import pygad
import random
import matplotlib.pyplot as plt
from dd.autoref import BDD


def find_predictors_to_invert(data, predictors):
    # обучаем логистическую регрессию с выделенными признаками,
    #   выбираем признаки с отрицательными весами
    data.prepare(predictors, "Dead", [])
    logist_reg = LogisticRegression()
    logist_reg.fit(data.x, data.y)
    weights = logist_reg.coef_.ravel()
    invert_predictors = []
    for i, feature_name in enumerate(predictors):
        if weights[i] < 0:
            invert_predictors.append(feature_name)
    return invert_predictors


def print_model(model, data):
    print("=" * 10 + "\nМодель")
    print("Пороги:")
    for k, feature_name in enumerate(predictors):
        val = data.get_coord(feature_name, model.cutoffs[k])
        s = '≤' if feature_name in data.inverted_predictors else '≥'
        print(feature_name, " ", s, val, sep='')


def t_model(model, x_test, y_test, p_threshold):
    p = model.predict_proba(x_test)[:, 1]
    auc = sklearn_metrics.roc_auc_score(y_test, p)
    print("AUC:", auc)
    # выводим качество модели
    y_pred = np.where(p > p_threshold, 1, 0)
    tn, fp, fn, tp = sklearn_metrics.confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print("Sens:", sensitivity, "Spec:", specificity)
    print("tp =", tp, "fn =", fn, "fp =", fp, "tn =", tn)
    return auc, sensitivity, specificity


def plot_step(x1, x1_name, x2, x2_name, y, cutoffs1, cutoffs2):
   # min_x1 = np.min(x1)
   # min_x2 = np.min(x2)
   # max_x1 = np.max(x1)
   # max_x2 = np.max(x2)

    val_x1 = np.array([data.get_coord(x1_name, x1[i]) for i in range(x1.shape[0])])
    val_x2 = np.array([data.get_coord(x2_name, x2[i]) for i in range(x2.shape[0])])

    max_val_x1 = np.max(val_x1)
    max_val_x2 = np.max(val_x2)

    plt.scatter(val_x1[y == 0], val_x2[y == 0], marker='.', c='blue')  #, alpha=0.5)  #, linewidths=1)
    plt.scatter(val_x1[y == 1], val_x2[y == 1], marker='x', c='red')  #alpha=0.5

    """
    for v in cutoffs1:
        plt.axline((min_x1, v), (max_x1, v), c='green')
    for v in cutoffs2:
        plt.axline((v, min_x2), (v, max_x2), c='green')
    """

    print(cutoffs1)
    print(cutoffs2)
    print(max_val_x1, max_val_x2)

    #plt.axline((cutoffs1[-1], cutoffs2[0]), (max_val_x1, cutoffs2[0]), c='green')


    plt.plot([cutoffs1[4], max_val_x1], [cutoffs2[0], cutoffs2[0]], c='green', linewidth=3)
    plt.plot([cutoffs1[4], cutoffs1[4]], [cutoffs2[0], cutoffs2[1]], c='green', linewidth=3)

    plt.plot([cutoffs1[3], cutoffs1[4]], [cutoffs2[1], cutoffs2[1]], c='green', linewidth=3)
    plt.plot([cutoffs1[3], cutoffs1[3]], [cutoffs2[1], cutoffs2[2]], c='green', linewidth=3)

    plt.plot([cutoffs1[2], cutoffs1[3]], [cutoffs2[2], cutoffs2[2]], c='green', linewidth=3)
    plt.plot([cutoffs1[2], cutoffs1[2]], [cutoffs2[2], cutoffs2[3]], c='green', linewidth=3)

    plt.plot([cutoffs1[1], cutoffs1[2]], [cutoffs2[3], cutoffs2[3]], c='green', linewidth=3)
    plt.plot([cutoffs1[1], cutoffs1[1]], [cutoffs2[3], cutoffs2[4]], c='green', linewidth=3)

    plt.plot([cutoffs1[0], cutoffs1[1]], [cutoffs2[4], cutoffs2[4]], c='green', linewidth=3)
    plt.plot([cutoffs1[0], cutoffs1[0]], [cutoffs2[4], max_val_x2], c='green', linewidth=3)

    plt.xlabel(x1_name)
    plt.ylabel(x2_name)

    plt.show()


data = Data("DataSet_new.xlsx")

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]

#predictors = ["Age", "Killip class", "Cr", "EF LV", "NEUT"]

"""
threshold = 0.15:
    (Age & Cr & EF_LV & Glu & NEUT) | (Age & Cr & EF_LV & Killip_class & NEUT) | (Age & Cr & Glu & HR & NEUT) | (Age & Cr & Glu & Killip_class & NEUT) | (Age & Cr & Glu & NEUT & SBP) | (Cr & EF_LV & EOS & Killip_class & SBP) | (Cr & EF_LV & EOS & NEUT & SBP) | (Cr & EOS & HR & Killip_class & SBP) | (EF_LV & HR & Killip_class & NEUT & SBP) 
"""

"""
predictors = ["Age", "Cr", "EF LV", "NEUT", "Glu"]
Age ≥69.46343099746149
Cr ≥167.944027130994
EF LV ≤33.033174188660354
NEUT ≥75.42117692704959
Glu ≥7.062776779828779
EF_LV | (Age & Cr) | (Age & NEUT) | (Cr & Glu) | (Cr & NEUT) | (Glu & NEUT)

predictors = ["Age", "Killip class", "Cr", "EF LV", "NEUT"]
Age ≥70.57561758066473
Killip class ≥2.9698577158827306
Cr ≥154.18287069093304
EF LV ≤35.37571892147048
NEUT ≥75.21380169168862
(Age & NEUT) | (Cr & NEUT) | (EF_LV & NEUT) | (Killip_class & NEUT) | (Age & Cr & ~Killip_class) | (Age & EF_LV & ~Killip_class) | (Age & Killip_class & ~Cr) | (Cr & EF_LV & ~Age) | (Cr & Killip_class & ~Age) | (EF_LV & Killip_class & ~Age)

predictors = ["Age", "HR", "Cr", "NEUT", "Glu"]
Age ≥68.40573101284396
HR ≥84.51792574832406
Cr ≥159.83739760787995
NEUT ≥74.99352559492017
Glu ≥12.605976001915433
(Age & NEUT) | (Cr & NEUT) | (Glu & NEUT) | (HR & NEUT) | (Cr & Glu & ~HR) | (Cr & HR & ~Glu) | (Glu & HR & ~Age & ~Cr)
"""

#set_cutoffs = None

#set_cutoffs = [70, 82, 3, 135.7, 45, 75.6, 0.48, 0.24, 6.83, 115]

"""
На новых данных:
Модель
Пороги:
Age ≥68.4167108422245
HR ≥82.66553260745121
Killip class ≥2.9149268036634406
Cr ≥128.25045963014045
EF LV ≤47.70503668713291
NEUT ≥74.46746039457169
EOS ≤0.53328754140923
PCT ≥0.3883761527562328
Glu ≥10.579654694751333
SBP ≤126.7127359974053
"""

simplified_aggregation = False

set_cutoffs = [68, 83, 3, 129, 47, 74, 0.53, 0.39, 10.58, 126]

invert_predictors = find_predictors_to_invert(data, predictors)
data.prepare(predictors, "Dead", invert_predictors)


set_all_cutoffs = [[60.974430441762586, 81.43714510034326, 2.9149268036634406, 96.45099908763994, 48.695076216766005,
                   75.0980117759417, 1.3090322242800854, 0.23939388108101103, 6.689632704202784, 114.9781078659088],
                   [66.13692832968316, 78.47766971087532, 2.9149268036634406, 108.16382636058083, 51.40655843722995,
                    79.0397352546504, 0.2309940496156302, 1.494736126513699, 8.70599952999044, 126.60773898390306],
                   [66.409914050555, 86.64546812977943, 2.9149268036634406, 154.52695043817903, 53.884937316250124,
                    78.90835631366298, 0.12934343382678182, 0.20411609232285877, 9.746436533407984, 129.00857087409457],
                   [71.467781842493, 83.7460881987596, 2.9149268036634406, 123.51432709312061, 52.65206212417899,
                    75.45526685294391, 0.4995557278518177, 0.2849261252096274, 11.18142762773943, 117.12820666794772],
                   [67.65836078376293, 88.48990910182567, 2.9149268036634406, 163.30844754102156, 47.351081439788516,
                    77.98504476589562, 0.5877022934324242, 0.21935814178525076, 9.846333421628303, 116.34833582007585]]




"""
# оставим только часть данных - для тестирования
data.x = data.x[:100, :]
data.y = data.y[:100]
print(data.x.shape, data.y.shape)
"""


"""
transformed_all_cutoffs = []
for l in set_all_cutoffs:
    tl = []
    for i, nt in enumerate(l):
        val_normal = nt
        if predictors[i] in invert_predictors:
            val_normal = -val_normal
        val = (val_normal - data.scaler_mean[i]) / data.scaler_scale[i]
        tl.append(val)
    transformed_all_cutoffs.append(tl)
set_all_cutoffs = np.array(transformed_all_cutoffs)
"""

set_all_cutoffs = np.array(set_all_cutoffs)

#print(set_all_cutoffs)


threshold = 0.025

num_splits = 2
random_state = 123

np.random.seed(random_state)
random.seed(random_state)



ind1 = predictors.index('Age')
ind2 = predictors.index('NEUT')

sorted1 = sorted(set_all_cutoffs[:, ind1])
sorted2 = sorted(set_all_cutoffs[:, ind2])
if predictors[ind1] in invert_predictors:
    sorted1 = sorted(sorted1, reverse=True)
if predictors[ind2] in invert_predictors:
    sorted2 = sorted(sorted2, reverse=True)

# TODO: разобраться со случаем инвертирования!
# Для тестирования можно добавить знак "минус" перед инвертированным предиктором.

plot_step(data.x[:, ind1], predictors[ind1], data.x[:, ind2], predictors[ind2], data.y[:], sorted1, sorted2)
