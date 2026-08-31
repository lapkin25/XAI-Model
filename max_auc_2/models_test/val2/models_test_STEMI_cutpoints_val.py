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

def save_sample_to_file(arr, file_name):
    with open(file_name, "w") as file:
        for x in arr:
            file.write(str(x) + "\n")


data = Data("STEMI.xlsx", STEMI=True)

predictors = ['Возраст', 'NER1', 'SIRI', 'СОЭ', 'TIMI после', 'СДЛА', 'Killip',
              'RR 600-1200', 'интервал PQ 120-200', 'NBR1', 'SII', 'EOS', 'NEUT']

data.prepare(predictors, "isAFAfter", [], scale_data=False)


random_state = 1234

x_train, x_test, y_train, y_test = \
    train_test_split(data.x, data.y, test_size=0.2, stratify=data.y,
                     random_state=random_state)  # закомментировать random_state



assert(data.x.shape[1] == len(predictors))

df = pd.DataFrame(columns=predictors)
for i, name in enumerate(predictors):
    df.loc[:, name] = x_test[:, i]  #data.x[:, i]
df.loc[:, 'isAFAfter'] = y_test[:]  #data.y[:]

print("Теперь ", len(df), " наблюдений, ", sum(y_test), "заболеваемость")


# Пороги:
# Возраст ≥65.93737748427554
# NER1 ≥2387.6009243842545
# SIRI ≥5.022349948362129
# СОЭ ≥29.6049917083199
# TIMI после ≤1.7075872835465769
# СДЛА ≥22.230746149212216
# Killip ≥3.0257566769725166
# RR 600-1200 ≤543.4873416057574
# интервал PQ 120-200 ≥218.96049246714693
# RR 600-1200_ ≥1336.2440263241833

# Фенотипы:
# Возраст & (СДЛА)
# NER1 & (СОЭ)
# SIRI
# СОЭ & (СДЛА)
# TIMI после
# СДЛА & (Возраст | SIRI | Killip | TIMI после)
# Killip & (СДЛА)
# RR 600-1200 & (Возраст | Killip)
# интервал PQ 120-200 & (SIRI | RR 600-1200)
# RR 600-1200_ & (Возраст)

# Веса: [ 0.34709376 -0.08150446  0.66487236  0.70630475  0.73151799  0.88766336
#   1.06919782  1.18283922  0.28616948  0.        ]



# Возраст & (СДЛА)
df.loc[(df['Возраст'] >= 65.9373) & ((df['СДЛА'] >= 22.2307)), ('F1')] = 1
df.loc[(df['Возраст'] < 65.9373) | ((df['СДЛА'] < 22.2307)), ('F1')] = 0

# SIRI
df.loc[(df['SIRI'] >= 5.0223), ('F3')] = 1
df.loc[(df['SIRI'] < 5.0223), ('F3')] = 0

# СОЭ & (СДЛА)
df.loc[(df['СОЭ'] >= 29.6049) & ((df['СДЛА'] >= 22.2307)), ('F4')] = 1
df.loc[(df['СОЭ'] < 29.6049) | ((df['СДЛА'] < 22.2307)), ('F4')] = 0

# TIMI после
df.loc[(df['TIMI после'] <= 1.707587), ('F5')] = 1
df.loc[(df['TIMI после'] > 1.707587), ('F5')] = 0

# СДЛА & (Возраст | SIRI | Killip | TIMI после)
df.loc[(df['СДЛА'] >= 22.2307) & ((df['Возраст'] >= 65.937) |
                                  (df['SIRI'] >= 5.0223) |
                                  (df['Killip'] >= 3.02575) |
                                  (df['TIMI после'] <= 1.707587)), ('F6')] = 1
df.loc[(df['СДЛА'] < 22.2307) | ((df['Возраст'] < 65.937) &
                                  (df['SIRI'] < 5.0223) &
                                  (df['Killip'] < 3.02575) &
                                  (df['TIMI после'] > 1.707587)), ('F6')] = 0

# Killip & (СДЛА)
df.loc[(df['Killip'] >= 3.02575) & ((df['СДЛА'] >= 22.2307)), ('F7')] = 1
df.loc[(df['Killip'] < 3.02575) | ((df['СДЛА'] < 22.2307)), ('F7')] = 0

# RR 600-1200 & (Возраст | Killip)
df.loc[(df['RR 600-1200'] <= 543.48734) & ((df['Возраст'] >= 65.937) |
                                           (df['Killip'] >= 3.02575)), ('F8')] = 1
df.loc[(df['RR 600-1200'] > 543.48734) | ((df['Возраст'] < 65.937) &
                                           (df['Killip'] < 3.02575)), ('F8')] = 0

# интервал PQ 120-200 & (SIRI | RR 600-1200)
df.loc[(df['интервал PQ 120-200'] >= 218.9604) & ((df['SIRI'] >= 5.0223) |
                                                  (df['RR 600-1200'] <= 543.48734)), ('F9')] = 1
df.loc[(df['интервал PQ 120-200'] < 218.9604) | ((df['SIRI'] < 5.0223) &
                                                  (df['RR 600-1200'] > 543.48734)), ('F9')] = 0


features = [ 'F1', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']


isModel = 1 # 1 - Logistic
rm = 100
border = 0.065  #0.08
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

