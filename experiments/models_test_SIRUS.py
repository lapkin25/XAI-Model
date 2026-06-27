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





data = Data("DataSet.xlsx")

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]

data.prepare(predictors, "Dead", [], scale_data=False)

assert(data.x.shape[1] == len(predictors))

df = pd.DataFrame(columns=predictors)
for i, name in enumerate(predictors):
    df.loc[:, name] = data.x[:, i]
df.loc[:, 'Dead'] = data.y[:]



df.loc[(df['Cr'] >= 150) & (df['HR'] >= 98), ('F1')] = 0.003
df.loc[(df['Cr'] < 150) | (df['HR'] < 98), ('F1')] = 0.001

df.loc[(df['SBP'] < 105) & (df['HR'] >= 98), ('F2')] = 0.003
df.loc[(df['SBP'] >= 105) | (df['HR'] < 98), ('F2')] = 0.001

df.loc[(df['EF LV'] < 42) & (df['SBP'] < 105), ('F3')] = 0.003
df.loc[(df['EF LV'] >= 42) | (df['SBP'] >= 105), ('F3')] = 0.001

df.loc[(df['HR'] >= 98) & (df['NEUT'] >= 83.2), ('F4')] = 0.004
df.loc[(df['HR'] < 98) | (df['NEUT'] < 83.2), ('F4')] = 0.001

df.loc[(df['Glu'] >= 10.09) & (df['SBP'] < 105), ('F5')] = 0.003
df.loc[(df['Glu'] < 10.09) | (df['SBP'] >= 105), ('F5')] = 0.001

df.loc[(df['EF LV'] < 42) & (df['Glu'] >= 10.09), ('F6')] = 0.002
df.loc[(df['EF LV'] >= 42) | (df['Glu'] < 10.09), ('F6')] = 0.001

df.loc[(df['EF LV'] < 42) & (df['NEUT'] >= 83.2), ('F7')] = 0.003
df.loc[(df['EF LV'] >= 42) | (df['NEUT'] < 83.2), ('F7')] = 0.001

df.loc[(df['Glu'] >= 7.89) & (df['SBP'] < 105), ('F8')] = 0.003
df.loc[(df['Glu'] < 7.89) | (df['SBP'] >= 105), ('F8')] = 0.001

df.loc[(df['Cr'] >= 150) & (df['NEUT'] >= 83.2), ('F9')] = 0.004
df.loc[(df['Cr'] < 150) | (df['NEUT'] < 83.2), ('F9')] = 0.001

df.loc[(df['Glu'] < 4.92) & (df['Cr'] >= 150), ('F10')] = 0.001
df.loc[(df['Glu'] >= 4.92) | (df['Cr'] < 150), ('F10')] = 0.0

features =[ 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10' ]


pred = df['F1'] + df['F2'] + df['F3'] + df['F4'] + df['F5'] + df['F6'] + df['F7'] + df['F8'] + df['F9'] + df['F10']

print(pred)

fpr, tpr, _ = roc_curve(data.y, pred)
roc_auc = auc(fpr, tpr)

print("AUC_SIRUS =", roc_auc)

isModel = 1 # 1 - Logistic
rm = 100
border = 0.04   # 0.03
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

    # Выборка из одного признака
    x_feat = np.array(df[feat]).reshape(-1, 1)

    #print(x_feat)

#    ct = pd.crosstab(y1, df[feat])
#    table = sm.stats.Table2x2(ct, shift_zeros=False)
#    print(table)
#    odds_ratio = table.oddsratio
#    confint = table.oddsratio_confint()

 #   print('Odds ratio, 95% CI', odds_ratio, confint)

    fpr, tpr, _ = roc_curve(y1, df[feat])
    roc_auc = auc(fpr, tpr)
    print("AUC =", roc_auc)
