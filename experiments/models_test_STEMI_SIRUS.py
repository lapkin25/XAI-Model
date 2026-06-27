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



data = Data("STEMI.xlsx", STEMI=True)

predictors = ['Возраст', 'NER1', 'SIRI', 'СОЭ', 'TIMI после', 'СДЛА', 'Killip',
              'RR 600-1200', 'интервал PQ 120-200', 'NBR1', 'SII']

data.prepare(predictors, "isAFAfter", [], scale_data=False)

assert(data.x.shape[1] == len(predictors))

df = pd.DataFrame(columns=predictors)
for i, name in enumerate(predictors):
    df.loc[:, name] = data.x[:, i]
df.loc[:, 'isAFAfter'] = data.y[:]


df.loc[(df['SIRI'] >= 5.3615384) & (df['NER1'] < 46.153847), ('F1')] = 0.001
df.loc[(df['SIRI'] < 5.3615384) | (df['NER1'] >= 46.153847), ('F1')] = 0.0

df.loc[(df['RR 600-1200'] >= 1040.0) & (df['TIMI после'] < 3), ('F2')] = 0.001
df.loc[(df['RR 600-1200'] < 1040.0) | (df['TIMI после'] >= 3), ('F2')] = 0.0

df.loc[(df['SIRI'] < 0.9692308) & (df['RR 600-1200'] < 600), ('F3')] = 0.004
df.loc[(df['SIRI'] >= 0.9692308) | (df['RR 600-1200'] >= 600), ('F3')] = 0.001

df.loc[(df['TIMI после'] < 2) & (df['RR 600-1200'] < 600), ('F4')] = 0.004
df.loc[(df['TIMI после'] >= 2) | (df['RR 600-1200'] >= 600), ('F4')] = 0.001

df.loc[(df['NER1'] >= 281.2) & (df['RR 600-1200'] >= 1040), ('F5')] = 0.007
df.loc[(df['NER1'] < 281.2) | (df['RR 600-1200'] < 1040), ('F5')] = 0.002

df.loc[(df['NER1'] >= 281.2) & (df['TIMI после'] < 2), ('F6')] = 0.007
df.loc[(df['NER1'] < 281.2) | (df['TIMI после'] >= 2), ('F6')] = 0.002

df.loc[(df['SIRI'] < 0.9692308) & (df['NER1'] >= 170), ('F7')] = 0.003
df.loc[(df['SIRI'] >= 0.9692308) | (df['NER1'] < 170), ('F7')] = 0.001

df.loc[(df['NER1'] >= 725.5) & (df['TIMI после'] < 2), ('F8')] = 0.005
df.loc[(df['NER1'] < 725.5) | (df['TIMI после'] >= 2), ('F8')] = 0.002

df.loc[(df['интервал PQ 120-200'] < 140) & (df['СОЭ'] >= 46), ('F9')] = 0.004
df.loc[(df['интервал PQ 120-200'] >= 140) | (df['СОЭ'] < 46), ('F9')] = 0.001

df.loc[(df['NER1'] >= 725.5) & (df['TIMI после'] < 3), ('F10')] = 0.008
df.loc[(df['NER1'] < 725.5) | (df['TIMI после'] >= 3), ('F10')] = 0.003


features = [ 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10' ]


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
y1 = np.array(df['isAFAfter'].astype('int'))
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
