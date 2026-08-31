import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score, confusion_matrix, f1_score, accuracy_score, recall_score, \
    roc_curve, auc
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold

sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data
import pandas as pd
import numpy as np
import statsmodels.api as sm
import xgboost as xgb
from catboost import CatBoostClassifier


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


data = Data("DataSet_val.xlsx")

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]

data.prepare(predictors, "Dead", [], scale_data=False)

assert(data.x.shape[1] == len(predictors))

df = pd.DataFrame(columns=predictors)
for i, name in enumerate(predictors):
    df.loc[:, name] = data.x[:, i]
df.loc[:, 'Dead'] = data.y[:]


# Age ≥51.0
#   HR ≥80.6
df.loc[(df['Age'] >= 51.0) & (df['HR'] >= 80.6), ('F1')] = 1
df.loc[(df['Age'] < 51.0) | (df['HR'] < 80.6), ('F1')] = 0

# Age ≥51.0
#   Killip class ≥3.0
df.loc[(df['Age'] >= 51.0) & (df['Killip class'] >= 3.0), ('F2')] = 1
df.loc[(df['Age'] < 51.0) | (df['Killip class'] < 3.0), ('F2')] = 0

#   Age ≥49.7
#   Cr ≥120.767
df.loc[(df['Age'] >= 49.7) & (df['Cr'] >= 120.767), ('F3')] = 1
df.loc[(df['Age'] < 49.7) | (df['Cr'] < 120.767), ('F3')] = 0

#  Age ≥51.0
#   NEUT ≥74.94
df.loc[(df['Age'] >= 51.0) & (df['NEUT'] >= 74.94), ('F4')] = 1
df.loc[(df['Age'] < 51.0) | (df['NEUT'] < 74.94), ('F4')] = 0

#Age ≥55.2
#  EOS ≤0.6200000000000001
df.loc[(df['Age'] >= 55.2) & (df['EOS'] <= 0.6200000000000001), ('F5')] = 1
df.loc[(df['Age'] < 55.2) | (df['EOS'] > 0.6200000000000001), ('F5')] = 0

#   HR ≥87.1
#   PCT ≥0.251
df.loc[(df['HR'] >= 87.1) & (df['PCT'] >= 0.251), ('F6')] = 1
df.loc[(df['HR'] < 87.1) | (df['PCT'] < 0.251), ('F6')] = 0

#   HR ≥80.0
#   Glu ≥6.205
df.loc[(df['HR'] >= 80.0) & (df['Glu'] >= 6.205), ('F7')] = 1
df.loc[(df['HR'] < 80.0) | (df['Glu'] < 6.205), ('F7')] = 0

#   EF LV ≤50.0
#   EOS ≤0.96
df.loc[(df['EF LV'] <= 50.0) & (df['EOS'] <= 0.96), ('F8')] = 1
df.loc[(df['EF LV'] > 50.0) | (df['EOS'] > 0.96), ('F8')] = 0

#   EF LV ≤50.0
#   SBP ≤115.0
df.loc[(df['EF LV'] <= 50.0) & (df['SBP'] <= 115.0), ('F9')] = 1
df.loc[(df['EF LV'] > 50.0) | (df['SBP'] > 115.0), ('F9')] = 0

#   EOS ≤0.81
#   Glu ≥6.343
df.loc[(df['EOS'] <= 0.81) & (df['Glu'] >= 6.343), ('F10')] = 1
df.loc[(df['EOS'] > 0.81) | (df['Glu'] < 6.343), ('F10')] = 0



features =[ 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10' ]

isModel = 1 # 1 - Logistic
rm = 100
border = 0.04 #0.03
np.random.seed(rm)

x_all = np.array(df[features], dtype=int)
y1 = np.array(df['Dead'].astype('int'))
#y_all = utils.to_categorical(y1)

# Параметры для модели
solver1 = 'lbfgs'
max_iter1 = 2000
C1 = 1
penalty1 = 'l2'

lr = 0.1
m_d = 2
n_e = 100
spw = 1

m_d1 = 3
n_e1 = 180

# оцениваем точность каждого фактора риска
for feat in features:
    print("ФР", feat)

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
    x_feat = np.array(df[feat]).reshape(-1, 1)

    ct = pd.crosstab(y1, df[feat])
    table = sm.stats.Table2x2(ct, shift_zeros=False)
    print(table)
    odds_ratio = table.oddsratio
    confint = table.oddsratio_confint()

    print('Odds ratio, 95% CI', odds_ratio, confint)

    # x_feat, y1
    x_train = x_feat
    y_train = y1

    model = LogisticRegression(solver=solver1, max_iter=max_iter1, C=C1, penalty=penalty1)
    # Настроим метрики для кросс-валидации
    scoring = {'roc_auc': make_scorer(roc_auc_score, response_method='predict_proba'),
               'f1': make_scorer(custom_f1_score, response_method='predict_proba', threshold=border),
               'accuracy': make_scorer(custom_accuracy_score, response_method='predict_proba', threshold=border),
               'sensitivity': make_scorer(custom_recall_score, response_method='predict_proba', threshold=border),
               'specificity': make_scorer(custom_specificity_score, response_method='predict_proba', threshold=border)
               }

    model.fit(x_train, y_train)
    y_pred_prob = model.predict_proba(x_train)[:, 1]  # Вероятности для положительного класса
    y_pred = (y_pred_prob >= border).astype(int)
    fpr, tpr, _ = roc_curve(y_train, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC: {roc_auc:.4f}")