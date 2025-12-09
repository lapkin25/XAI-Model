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


data = Data("DataSet.xlsx")

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]

data.prepare(predictors, "Dead", [], scale_data=False)

assert(data.x.shape[1] == len(predictors))

df = pd.DataFrame(columns=predictors)
for i, name in enumerate(predictors):
    df.loc[:, name] = data.x[:, i]
df.loc[:, 'Dead'] = data.y[:]

df.loc[:, ('F1')] = 0
Age = (df['Age'] >= 63)
HR = (df['HR'] >= 83)
NEUT = (df['NEUT'] >= 74)
EOS = (df['EOS'] <= 0.31)
Glu = (df['Glu'] >= 9.3)
df.loc[(Age & NEUT) | (EOS & Glu) | (Glu & HR) | (HR & NEUT) | (Age & EOS & HR), ('F1')] = 1


df.loc[:, ('F2')] = 0
Age = (df['Age'] >= 77)
HR = (df['HR'] >= 86)
EOS = (df['EOS'] <= 0.33)
Glu = (df['Glu'] >= 7.4)
df.loc[(Age & EOS) | (Age & Glu) | (EOS & Glu) | (EOS & HR) | (Glu & HR), ('F2')] = 1


df.loc[:, ('F3')] = 0
Age = (df['Age'] >= 63)
HR = (df['HR'] >= 76)
EOS = (df['EOS'] <= 0.2)
NEUT = (df['NEUT'] >= 75.2)
Killip_class = (df['Killip class'] >= 3)
df.loc[(EOS & NEUT) | (Killip_class & NEUT) | (Age & EOS & HR) | (Age & HR & Killip_class) | (Age & HR & NEUT), ('F3')] = 1


df.loc[:, ('F4')] = 0
Age = (df['Age'] >= 65)
PCT = (df['PCT'] >= 0.28)
Glu = (df['Glu'] >= 6.71)
df.loc[Glu | (Age & PCT), ('F4')] = 1


df.loc[:, ('F5')] = 0
Killip_class = (df['Killip class'] >= 3)
Glu = (df['Glu'] >= 8.41)
NEUT = (df['NEUT'] >= 75.2)
EOS = (df['EOS'] <= 0.18)
HR = (df['HR'] >= 84)
df.loc[(EOS & HR) | (Glu & Killip_class) | (Glu & NEUT) | (HR & NEUT) | (Killip_class & NEUT), ('F5')] = 1


df.loc[:, ('F6')] = 0
HR = (df['HR'] >= 85)
SBP = (df['SBP'] <= 114)
Glu = (df['Glu'] >= 7.31)
EOS = (df['EOS'] <= 0.58)
NEUT = (df['NEUT'] >= 75.2)
df.loc[(EOS & HR) | (EOS & NEUT) | (Glu & HR) | (HR & SBP), ('F6')] = 1


df.loc[:, ('F7')] = 0
PCT = (df['PCT'] >= 0.31)
HR = (df['HR'] >= 84)
Glu = (df['Glu'] >= 7.52)
EOS = (df['EOS'] <= 0.24)
df.loc[EOS | (Glu & HR) | (Glu & PCT) | (HR & PCT), ('F7')] = 1


df.loc[:, ('F8')] = 0
HR = (df['HR'] >= 86)
Cr = (df['Cr'] >= 172)
Glu = (df['Glu'] >= 7.12)
EOS = (df['EOS'] <= 0.28)
df.loc[(Cr & EOS) | (Cr & Glu) | (EOS & Glu) | (EOS & HR) | (Glu & HR), ('F8')] = 1


df.loc[:, ('F9')] = 0
HR = (df['HR'] >= 84)
EOS = (df['EOS'] <= 0.33)
Glu = (df['Glu'] >= 7.5)
df.loc[(EOS & Glu) | (EOS & HR) | (Glu & HR), ('F9')] = 1


df.loc[:, ('F10')] = 0
HR = (df['HR'] >= 79)
EOS = (df['EOS'] <= 0.16)
NEUT = (df['NEUT'] >= 76.7)
Glu = (df['Glu'] >= 8.9)
df.loc[NEUT | (EOS & HR) | (Glu & HR), ('F10')] = 1


df.loc[:, ('F11')] = 0
Cr = (df['Cr'] >= 140)
PCT = (df['PCT'] >= 0.33)
SBP = (df['SBP'] <= 99)
df.loc[Cr | PCT | SBP, ('F11')] = 1


df.loc[:, ('F12')] = 0
EF_LV = (df['EF LV'] <= 45)
Glu = (df['Glu'] >= 7.2)
HR = (df['HR'] >= 85)
df.loc[EF_LV | (Glu & HR), ('F12')] = 1


features = [ 'F1', 'F2', 'F3', 'F4',  'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12' ]


isModel = 1 # 1 - Logistic
rm = 100
border = 0.03
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
