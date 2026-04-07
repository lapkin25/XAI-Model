import sys

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data
from sklearn.linear_model import LogisticRegression
import csv
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, roc_auc_score, confusion_matrix, f1_score, accuracy_score, recall_score, \
    roc_curve, auc
import sklearn.metrics as sklearn_metrics
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
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


# Функция для вычисления 95% доверительного интервала
def compute_confidence_interval(data):
    mean = np.mean(data)
    std_error = np.std(data, ddof=1) / np.sqrt(len(data))
    confidence_interval = 1.96 * std_error
    return mean, mean - confidence_interval, mean + confidence_interval


threshold = 0.05

# Функция для расчета специфичности
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

# Функция для преобразования вероятностей в бинарные метки с заданным порогом
def custom_predict(proba, threshold=threshold):  # Устанавливаем порог 0.05
    return (proba >= threshold).astype(int)

# Пользовательская функция для расчета метрик с учетом порога
def custom_metric(y_true, proba, metric_func, threshold=threshold):  # Устанавливаем порог 0.05
    y_pred = custom_predict(proba, threshold)
    return metric_func(y_true, y_pred)

# Создаем пользовательские scorer'ы с порогом 0.05
def custom_f1_score(y_true, proba, threshold=threshold):
    return custom_metric(y_true, proba, f1_score, threshold=threshold)

def custom_accuracy_score(y_true, proba, threshold=threshold):
    return custom_metric(y_true, proba, accuracy_score, threshold=threshold)

def custom_recall_score(y_true, proba, threshold=threshold):
    return custom_metric(y_true, proba, recall_score, threshold=threshold)

def custom_specificity_score(y_true, proba, threshold=threshold):
    return custom_metric(y_true, proba, specificity_score, threshold=threshold)



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


def plot_step(x1, x1_name, x2, x2_name, y, cutoffs1_, cutoffs2_):
   # min_x1 = np.min(x1)
   # min_x2 = np.min(x2)
   # max_x1 = np.max(x1)
   # max_x2 = np.max(x2)


    val_x1 = np.array([data.get_coord(x1_name, x1[i]) for i in range(x1.shape[0])])
    val_x2 = np.array([data.get_coord(x2_name, x2[i]) for i in range(x2.shape[0])])
    cutoffs1 = np.array(cutoffs1_)
    cutoffs2 = np.array(cutoffs2_)

    if x1_name in invert_predictors:
        val_x1 = -val_x1
        cutoffs1 = -cutoffs1
    if x2_name in invert_predictors:
        val_x2 = -val_x2
        cutoffs2 = -cutoffs2


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

    #plt.savefig("Fig1.png", dpi=300)
    #plt.savefig("Fig1.tiff", dpi=300)

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

#set_cutoffs = [68, 83, 3, 129, 47, 74, 0.53, 0.39, 10.58, 126]

invert_predictors = find_predictors_to_invert(data, predictors)
data.prepare(predictors, "Dead", invert_predictors)

"""
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

"""
МОДЕЛИ:

Модель 1:
==========
Модель
Пороги:
Age ?67.72801971709698
HR ?85.58356797650632
Killip class ?2.9149268036634406
Cr ?140.5573010066018
EF LV ?51.24715831952078
NEUT ?79.49696793759162
EOS ?0.43328357001971285
PCT ?0.26989146857550783
Glu ?7.819084358033083
SBP ?125.82246953030287

Модель 2:
==========
Модель
Пороги:
Age ?66.52761309865195
HR ?82.32364565085969
Killip class ?2.9149268036634406
Cr ?127.77519898251836
EF LV ?52.663708904133806
NEUT ?76.68352450240539
EOS ?0.24855461667427514
PCT ?0.1877433363316367
Glu ?6.681109899827095
SBP ?118.0169726517416

Модель 3:
==========
Модель
Пороги:
Age ?67.6844456158494
HR ?88.60251364321226
Killip class ?2.9149268036634406
Cr ?139.71837487391204
EF LV ?51.64700056055864
NEUT ?75.39091412245877
EOS ?1.03284344102116
PCT ?0.20858126820062853
Glu ?8.291353846228906
SBP ?138.48083273258095

Модель 4:
==========
Модель
Пороги:
Age ?68.5171458080733
HR ?87.9437317343314
Killip class ?2.9149268036634406
Cr ?104.79405620662472
EF LV ?49.22886428126665
NEUT ?76.64974071231322
EOS ?1.0929270141860519
PCT ?1.8443873785430864
Glu ?7.241579136959278
SBP ?130.35414769563423

Модель 5:
==========
Модель
Пороги:
Age ?68.2427506845099
HR ?86.08447702106446
Killip class ?2.9149268036634406
Cr ?142.3963987077279
EF LV ?46.99298802806198
NEUT ?76.07287634248969
EOS ?0.39510176027553645
PCT ?0.22485477434992357
Glu ?7.789882830490346
SBP ?131.71045201390467
"""


set_all_cutoffs = [
    [67.72801971709698, 85.58356797650632, 2.9149268036634406, 140.5573010066018, 51.24715831952078,
     79.49696793759162, 0.43328357001971285, 0.26989146857550783, 7.819084358033083, 125.82246953030287],
    [66.52761309865195, 82.32364565085969, 2.9149268036634406, 127.77519898251836, 52.663708904133806,
     76.68352450240539, 0.24855461667427514, 0.1877433363316367, 6.681109899827095, 118.0169726517416],
    [67.6844456158494, 88.60251364321226, 2.9149268036634406, 139.71837487391204, 51.64700056055864,
     75.39091412245877, 1.03284344102116, 0.20858126820062853, 8.291353846228906, 138.48083273258095],
    [68.5171458080733, 87.9437317343314, 2.9149268036634406, 104.79405620662472, 49.22886428126665,
     76.64974071231322, 1.0929270141860519, 1.8443873785430864, 7.241579136959278, 130.35414769563423],
    [68.2427506845099, 86.08447702106446, 2.9149268036634406, 142.3963987077279, 46.99298802806198,
     76.07287634248969, 0.39510176027553645, 0.22485477434992357, 7.789882830490346, 131.71045201390467]
]

set_all_cutoffs = np.array(set_all_cutoffs)


fav = [('EOS', 'Glu'), ('Age', 'NEUT'), ('HR', 'EOS'), ('EF LV', 'Glu'),
       ('Cr', 'NEUT'), ('Age', 'EOS'), ('HR', 'EF LV'), ('HR', 'Glu'),
       ('HR', 'Killip class'), ('EOS', 'SBP'), ('Killip class', 'EF LV'),
       ('Cr', 'Glu'), ('Cr', 'EOS'), ('Cr', 'EF LV'), ('NEUT', 'PCT')]
       #('EF LV', 'SBP'), ('Glu', 'SBP'), ('Cr', 'SBP'), ('Killip class', 'Glu')]
       #('Killip class', 'Cr'), ('HR', 'Cr'), ('Age', 'Glu'), ('HR', 'SBP')]

# вычисляем логические переменные (парные фенотипы)

phenotypes = []
for j in range(data.x.shape[1]):
    for k in range(j + 1, data.x.shape[1]):
        if (predictors[j], predictors[k]) not in fav:
            continue

        F = np.zeros(data.x.shape[0], dtype=int)
        for i in range(data.x.shape[0]):  # перечисляем все наблюдения
            def count_balls(p, val):
                """
                p - индекс предиктора
                """
                balls = 0  # сколько баллов дает p-й предиктор
                # считаем, сколько моделей считают p-й предиктор фактором риска
                for m in range(5):
                    if predictors[p] in invert_predictors:
                        if val <= set_all_cutoffs[m, p]:
                            balls += 1
                    else:
                        if val >= set_all_cutoffs[m, p]:
                            balls += 1
                return balls

            balls_j = count_balls(j, data.get_coord(predictors[j], data.x[i, j]))
            balls_k = count_balls(k, data.get_coord(predictors[k], data.x[i, k]))
            if balls_j + balls_k >= 5:
                F[i] = 1

            """
            balls_j = 0  # сколько баллов дает j-й предиктор
            # считаем, сколько моделей считают j-й предиктор фактором риска
            if predictors[j] in invert_predictors:
                
            balls_k = 0  # сколько баллов дает k-й предиктор
            # считаем, сколько моделей считают k-й предиктор фактором риска
            """

        phenotypes.append({'j': j, 'k': k, 'F': F})

#print(phenotypes)

phenotypes_x = np.zeros((data.x.shape[0], len(phenotypes)), dtype=int)
for p in range(len(phenotypes)):
    phenotypes_x[:, p] = phenotypes[p]['F']



isModel = 1 # 1 - Logistic
rm = 100
border = 0.03
np.random.seed(rm)

x_all = phenotypes_x  #np.array(df[features], dtype=int)
y1 = data.y   #np.array(df['Dead'].astype('int'))
#y_all = utils.to_categorical(y1)

# Параметры для модели
#solver1 = 'lbfgs'
solver1 = 'liblinear'
max_iter1 = 2000
C1 = 1
#penalty1 = 'l2'
penalty1 = 'l1'

lr = 0.1
m_d = 2
n_e = 100
spw = 1

m_d1 = 3
n_e1 = 180

"""

# оцениваем точность каждого фактора риска
for p in range(len(phenotypes)):
    print("ФР", predictors[phenotypes[p]['j']], '&', predictors[phenotypes[p]['k']])

    # Для хранения результатов
    mean_roc_auc_test = []
    mean_sen_test = []
    mean_spec_test = []
    mean_f1_test = []
    mean_acc_test = []
    mean_ppv_test = []
    mean_npv_test = []

    mean_roc_auc = []
    mean_sensitivity = []
    mean_specificity = []
    mean_acc = []
    mean_f1 = []
    mean_ppv = []
    mean_npv = []

    # Выборка из одного признака
    x_feat = phenotypes_x[:, p].reshape(-1, 1) #np.array(df[feat]).reshape(-1, 1)

    ct = pd.crosstab(y1, phenotypes_x[:, p])
    table = sm.stats.Table2x2(ct, shift_zeros=False)
    print(table)
    odds_ratio = table.oddsratio
    confint = table.oddsratio_confint()

    print('Odds ratio, 95% CI', odds_ratio, confint)

    # Повторить 10 раз
    for j in range(100):
        #print("SPLIT #", j)
        np.random.seed(j + 42)

        # Случайное разделение данных на обучающую (80%) и тестовую (20%) выборки
        x_train, x_validate, y_train, y_validate = train_test_split(x_feat, y1, train_size=0.8, stratify=y1,
                                                                    random_state=j + 42)
        model = LogisticRegression(solver=solver1, max_iter=max_iter1, C=C1, penalty=penalty1)
        # Настроим метрики для кросс-валидации
        scoring = {'roc_auc': make_scorer(roc_auc_score, response_method='predict_proba'),
                   'f1': make_scorer(custom_f1_score, response_method='predict_proba', threshold=border),
                   'accuracy': make_scorer(custom_accuracy_score, response_method='predict_proba', threshold=border),
                   'sensitivity': make_scorer(custom_recall_score, response_method='predict_proba', threshold=border),
                   'specificity': make_scorer(custom_specificity_score, response_method='predict_proba', threshold=border)
                   }
        # Выполним кросс-валидацию с использованием cross_validate
        cv_results = cross_validate(model, x_train, y_train, cv=StratifiedKFold(n_splits=10),
                                    scoring=scoring, return_train_score=False)
        # Сохранение метрик кросс-валидации
        mean_roc_auc.append(np.mean(cv_results['test_roc_auc']))
        mean_f1.append(np.mean(cv_results['test_f1']))
        mean_acc.append(np.mean(cv_results['test_accuracy']))
        mean_sensitivity.append(np.mean(cv_results['test_sensitivity']))
        mean_specificity.append(np.mean(cv_results['test_specificity']))

        # Обучение модели на всей обучающей выборке
        model.fit(x_train, y_train)
        # Тестирование на валидационной выборке (20%)
        y_pred_prob = model.predict_proba(x_validate)[:, 1]  # Вероятности для положительного класса
        y_pred = (y_pred_prob >= border).astype(int)
        # Матрица ошибок
        cMatrix = confusion_matrix(y_validate, y_pred)
        sensivity = cMatrix[1][1] / (cMatrix[1][0] + cMatrix[1][1])
        specifity = cMatrix[0][0] / (cMatrix[0][0] + cMatrix[0][1])
        fpr, tpr, _ = roc_curve(y_validate, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(y_validate, y_pred)
        acc = accuracy_score(y_validate, y_pred)
        ppv = cMatrix[1][1] / (cMatrix[1][1] + cMatrix[0][1])
        npv = cMatrix[0][0] / (cMatrix[0][0] + cMatrix[1][0])

        # Сохранение результатов
        mean_roc_auc_test.append(roc_auc)
        mean_sen_test.append(sensivity)
        mean_spec_test.append(specifity)
        mean_f1_test.append(f1)
        mean_acc_test.append(acc)
        mean_ppv_test.append(ppv)
        mean_npv_test.append(npv)

    roc_auc_mean, roc_auc_lower, roc_auc_upper = compute_confidence_interval(mean_roc_auc)
    f1_mean, f1_lower, f1_upper = compute_confidence_interval(mean_f1)
    acc_mean, acc_lower, acc_upper = compute_confidence_interval(mean_acc)
    sen_mean, sen_lower, sen_upper = compute_confidence_interval(mean_sensitivity)
    spec_mean, spec_lower, spec_upper = compute_confidence_interval(mean_specificity)

    # Вывод результатов
    print(f"Кросс-валидационный ROC-AUC: {roc_auc_mean:.4f} 95% [ {roc_auc_lower:.4f}, {roc_auc_upper:.4f} ]")
    #print(f"Кросс-валидация Чувствительность: {sen_mean:.4f} 95% [ {sen_lower:.4f}, {sen_upper:.4f} ]")
    #print(f"Кросс-валидация Специфичность: {spec_mean:.4f} 95% [ {spec_lower:.4f}, {spec_upper:.4f} ]")
    #print(f"Кросс-валидационный F1: {f1_mean:.4f} 95% [ {f1_lower:.4f}, {f1_upper:.4f} ]")
    #print(f"Кросс-валидационный (Accuracy): {acc_mean:.4f} 95% [ {acc_lower:.4f}, {acc_upper:.4f} ]")

    # Итоговые результаты с доверительными интервалами
    roc_auc_mean, roc_auc_lower, roc_auc_upper = compute_confidence_interval(mean_roc_auc_test)
    sen_mean, sen_lower, sen_upper = compute_confidence_interval(mean_sen_test)
    spec_mean, spec_lower, spec_upper = compute_confidence_interval(mean_spec_test)
    f1_mean, f1_lower, f1_upper = compute_confidence_interval(mean_f1_test)
    acc_mean, acc_lower, acc_upper = compute_confidence_interval(mean_acc_test)
    ppv_mean, ppv_lower, ppv_upper = compute_confidence_interval(mean_ppv_test)
    npv_mean, npv_lower, npv_upper = compute_confidence_interval(mean_npv_test)

    # Вывод результатов
    # print("---------------------------------------")
    print(f"Средний ROC-AUC: {roc_auc_mean:.4f} 95% [ {roc_auc_lower:.4f}, {roc_auc_upper:.4f} ]")
    #print(f"Средняя Чувствительность: {sen_mean:.4f} 95% [ {sen_lower:.4f}, {sen_upper:.4f} ]")
    #print(f"Средняя Специфичность: {spec_mean:.4f} 95% [ {spec_lower:.4f}, {spec_upper:.4f} ]")
    #print(f"Средняя F1: {f1_mean:.4f} 95% [ {f1_lower:.4f}, {f1_upper:.4f} ]")
    #print(f"Средняя Точность (Accuracy): {acc_mean:.4f} 95% [ {acc_lower:.4f}, {acc_upper:.4f} ]")
    #print(f"Средний PPV: {ppv_mean:.4f} 95% [ {ppv_lower:.4f}, {ppv_upper:.4f} ]")
    #print(f"Средний NPV: {npv_mean:.4f} 95% [ {npv_lower:.4f}, {npv_upper:.4f} ]")

"""

# Для хранения результатов
mean_roc_auc_test = []
mean_sen_test = []
mean_spec_test = []
mean_f1_test = []
mean_acc_test = []
mean_ppv_test = []
mean_npv_test = []

mean_roc_auc=[]
mean_sensitivity=[]
mean_specificity=[]
mean_acc=[]
mean_f1=[]
mean_ppv=[]
mean_npv=[]

# Повторить 10 раз
for j in range(100):
    print("SPLIT #", j)
    np.random.seed(j + 42)

    # Случайное разделение данных на обучающую (80%) и тестовую (20%) выборки
    x_train, x_validate, y_train, y_validate = train_test_split(x_all, y1, train_size=0.8, stratify=y1,
                                                                random_state=j + 42)

    # Создаем классификатор в зависимости от выбора модели
    if isModel == 1:
        model = LogisticRegression(solver=solver1, max_iter=max_iter1, C=C1, penalty=penalty1)
    elif isModel == 2:
        model = xgb.XGBClassifier(learning_rate=lr, eval_metric="auc", scale_pos_weight=spw,
                                  max_depth=m_d, n_estimators=n_e, random_state=j + 42)
    elif isModel == 3:
        model = CatBoostClassifier(learning_rate=lr, eval_metric="AUC", scale_pos_weight=spw, max_depth=m_d,
                                   n_estimators=n_e,
                                   random_state=j + 42,
                                   cat_features=list(range(len(features))),  # Указываем категориальные колонки
                                   verbose=0)
    else:
        model = RandomForestClassifier(random_state=j + 42, n_estimators=n_e1, max_depth=m_d1)

    # Настроим метрики для кросс-валидации
    scoring = {'roc_auc': make_scorer(roc_auc_score, response_method='predict_proba'),
               'f1': make_scorer(custom_f1_score, response_method='predict_proba', threshold=border),
               'accuracy': make_scorer(custom_accuracy_score, response_method='predict_proba', threshold=border),
               'sensitivity': make_scorer(custom_recall_score, response_method='predict_proba', threshold=border),
               'specificity': make_scorer(custom_specificity_score, response_method='predict_proba', threshold=border)
               }

    # Выполним кросс-валидацию с использованием cross_validate
    cv_results = cross_validate(model, x_train, y_train, cv=StratifiedKFold(n_splits=10),
                                scoring=scoring, return_train_score=False)

    # Сохранение метрик кросс-валидации
    mean_roc_auc.append(np.mean(cv_results['test_roc_auc']))
    mean_f1.append(np.mean(cv_results['test_f1']))
    mean_acc.append(np.mean(cv_results['test_accuracy']))
    mean_sensitivity.append(np.mean(cv_results['test_sensitivity']))
    mean_specificity.append(np.mean(cv_results['test_specificity']))

    # Обучение модели на всей обучающей выборке
    model.fit(x_train, y_train)

    #print(model.coef_)

    # Тестирование на валидационной выборке (20%)
    y_pred_prob = model.predict_proba(x_validate)[:, 1]  # Вероятности для положительного класса
    y_pred = (y_pred_prob >= border).astype(int)

    # Матрица ошибок
    cMatrix = confusion_matrix(y_validate, y_pred)
    sensivity = cMatrix[1][1] / (cMatrix[1][0] + cMatrix[1][1])
    specifity = cMatrix[0][0] / (cMatrix[0][0] + cMatrix[0][1])
    fpr, tpr, _ = roc_curve(y_validate, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_validate, y_pred)
    acc = accuracy_score(y_validate, y_pred)
    ppv = cMatrix[1][1] / (cMatrix[1][1] + cMatrix[0][1])
    npv = cMatrix[0][0] / (cMatrix[0][0] + cMatrix[1][0])

    # Сохранение результатов
    mean_roc_auc_test.append(roc_auc)
    mean_sen_test.append(sensivity)
    mean_spec_test.append(specifity)
    mean_f1_test.append(f1)
    mean_acc_test.append(acc)
    mean_ppv_test.append(ppv)
    mean_npv_test.append(npv)

roc_auc_mean, roc_auc_lower, roc_auc_upper = compute_confidence_interval(mean_roc_auc)
f1_mean, f1_lower, f1_upper = compute_confidence_interval(mean_f1)
acc_mean, acc_lower, acc_upper = compute_confidence_interval(mean_acc)
sen_mean, sen_lower, sen_upper = compute_confidence_interval(mean_sensitivity)
spec_mean, spec_lower, spec_upper = compute_confidence_interval(mean_specificity)

# Вывод результатов
print(f"Кросс-валидационный ROC-AUC: {roc_auc_mean:.4f} 95% [ {roc_auc_lower:.4f}, {roc_auc_upper:.4f} ]")
print(f"Кросс-валидация Чувствительность: {sen_mean:.4f} 95% [ {sen_lower:.4f}, {sen_upper:.4f} ]")
print(f"Кросс-валидация Специфичность: {spec_mean:.4f} 95% [ {spec_lower:.4f}, {spec_upper:.4f} ]")
print(f"Кросс-валидационный F1: {f1_mean:.4f} 95% [ {f1_lower:.4f}, {f1_upper:.4f} ]")
print(f"Кросс-валидационный (Accuracy): {acc_mean:.4f} 95% [ {acc_lower:.4f}, {acc_upper:.4f} ]")

# Итоговые результаты с доверительными интервалами
roc_auc_mean, roc_auc_lower, roc_auc_upper = compute_confidence_interval(mean_roc_auc_test)
sen_mean, sen_lower, sen_upper = compute_confidence_interval(mean_sen_test)
spec_mean, spec_lower, spec_upper = compute_confidence_interval(mean_spec_test)
f1_mean, f1_lower, f1_upper = compute_confidence_interval(mean_f1_test)
acc_mean, acc_lower, acc_upper = compute_confidence_interval(mean_acc_test)
ppv_mean, ppv_lower, ppv_upper = compute_confidence_interval(mean_ppv_test)
npv_mean, npv_lower, npv_upper = compute_confidence_interval(mean_npv_test)

# Вывод результатов
# print("---------------------------------------")
print(f"Средний ROC-AUC: {roc_auc_mean:.4f} 95% [ {roc_auc_lower:.4f}, {roc_auc_upper:.4f} ]")
print(f"Средняя Чувствительность: {sen_mean:.4f} 95% [ {sen_lower:.4f}, {sen_upper:.4f} ]")
print(f"Средняя Специфичность: {spec_mean:.4f} 95% [ {spec_lower:.4f}, {spec_upper:.4f} ]")
print(f"Средняя F1: {f1_mean:.4f} 95% [ {f1_lower:.4f}, {f1_upper:.4f} ]")
print(f"Средняя Точность (Accuracy): {acc_mean:.4f} 95% [ {acc_lower:.4f}, {acc_upper:.4f} ]")
print(f"Средний PPV: {ppv_mean:.4f} 95% [ {ppv_lower:.4f}, {ppv_upper:.4f} ]")
print(f"Средний NPV: {npv_mean:.4f} 95% [ {npv_lower:.4f}, {npv_upper:.4f} ]")




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
