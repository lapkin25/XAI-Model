import sys
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data
from sklearn.linear_model import LogisticRegression
import csv
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import sklearn.metrics as sklearn_metrics

data_file = 'AF'  # 'M'


class MaxAUCPhenotypesModel:
    # Алгоритм:
    #   Для каждого предиктора делаем фенотип:
    #      Подбираем первичный порог и настраиваем вторичную модель
    #      из условия max AUC фенотипа

    # Первичный порог подбирается перебором на сетке

    # Алгоритм настройки вторичной модели:
    #   Вторичные пороги находятся из условия max AUC классификатора (на части выборки, где первичный ФР = 1)
    #   Из полученных вторичных ФР выбираются бинарные признаки, дизъюнкция которых, умноженная на первичный ФР,
    #     обеспечит max AUC фенотипа

    def __init__(self):
        self.logistic_model = None
        self.primary_thresholds = None
        self.phenotypes = None  # для каждого предиктора - список пар с этим предиктором, входящих в фенотип

    def get_phenotypes(self, x, a):
        # x - исходные данные, a - первичные пороги
        data_size, num_features = x.shape[0], x.shape[1]
        z = np.zeros((data_size, num_features), dtype=int)
        for j in range(num_features):
            if len(self.phenotypes[j]) == 0:
                z[:, j] = np.where(x[:, j] >= a[j], 1, 0)
            else:
                for k in range(len(self.phenotypes[j])):
                    z[:, j] |= np.where(x[:, self.phenotypes[j][k]['feature']] >= self.phenotypes[j][k]['threshold'], 1, 0)
                z[:, j] &= np.where(x[:, j] >= a[j], 1, 0)
        return z

    def fit(self, x, y):
        grid_size = 100
        data_size, num_features = x.shape[0], x.shape[1]
        self.phenotypes = [[] for _ in range(num_features)]
        self.primary_thresholds = np.zeros(num_features)
        for j in range(num_features):
            # j - номер фенотипа
            print("Phenotype", j + 1)
            # перебираем первичный порог на сетке
            grid = np.linspace(min_thresholds[j], np.max(x[:, j]), grid_size, endpoint=False)
            max_overall_auc = 0
            best_a_j = None
            best_phenotype = None
            for a_j in grid:
                y_pred_1 = np.where(x[:, j] >= a_j, 1, 0)
                #print([v for v in y_pred])
                try:
                    auc_1 = sklearn_metrics.roc_auc_score(y, y_pred_1)  # AUC для отдельного ФР
                except:
                    # если в выборке представлен только один класс
                    continue
                # находим вторичные пороги
                secondary_thresholds = [None] * num_features
                for k in range(num_features):
                    if k == j:
                        continue
                    grid2 = np.linspace(min_thresholds[k], np.max(x[:, k]), grid_size, endpoint=False)
                    max_auc_2 = 0
                    best_b_k = None
                    for b_k in grid2:
                        y_pred_2 = np.where(x[y_pred_1 == 1, k] >= b_k, 1, 0)
                        # AUC вторичного ФР после "срабатывания" первичного
                        try:
                            auc_2 = sklearn_metrics.roc_auc_score(y[y_pred_1 == 1], y_pred_2)
                        except:
                            # если в выборке представлен только один класс
                            continue
                        if auc_2 > max_auc_2:
                            max_auc_2 = auc_2
                            best_b_k = b_k
                    secondary_thresholds[k] = best_b_k
                #print(secondary_thresholds)
                ok = True
                for k in range(num_features):
                    if k != j and secondary_thresholds[k] is None:
                        ok = False
                if not ok:
                    continue
                # отбираем ФР во вторичную модель
                predictors_used = []
                current_auc = auc_1  # текущий достигнутый AUC фенотипа
                cur_y_pred = np.zeros_like(x[:, j], dtype=int)
                current_phenotype = []
                while True:
                    max_auc_2 = 0
                    best_k = None
                    for k in range(num_features):
                        if j == k or (k in predictors_used):
                            continue
                        # пробуем добавить k-й предиктор к фенотипу
                        new_y_pred = cur_y_pred | (y_pred_1 & np.where(x[:, k] >= secondary_thresholds[k], 1, 0))
                        auc_2 = sklearn_metrics.roc_auc_score(y, new_y_pred)  # считаем AUC фенотипа
                        if auc_2 > max_auc_2:
                            max_auc_2 = auc_2
                            best_k = k
                    if max_auc_2 > current_auc:
                        current_auc = max_auc_2
                        cur_y_pred = cur_y_pred | (y_pred_1 & np.where(x[:, best_k] >= secondary_thresholds[best_k], 1, 0))
                        current_phenotype.append({'feature': best_k, 'threshold': secondary_thresholds[best_k]})
                        predictors_used.append(best_k)
                    else:
                        break
                if current_auc > max_overall_auc:
                    max_overall_auc = current_auc
                    best_a_j = a_j
                    best_phenotype = current_phenotype
            self.primary_thresholds[j] = best_a_j
            self.phenotypes[j] = best_phenotype

        z = self.get_phenotypes(x, self.primary_thresholds)

        self.logistic_model = LogisticRegression(max_iter=10000)
        self.logistic_model.fit(z, y)

    def predict_proba(self, x):
        z = self.get_phenotypes(x, self.primary_thresholds)
        return self.logistic_model.predict_proba(z)


def find_predictors_to_invert(data, predictors):
    # обучаем логистическую регрессию с выделенными признаками,
    #   выбираем признаки с отрицательными весами
    if data_file == 'AF':
        data.prepare(predictors, "isAFAfter", [])
    else:
        data.prepare(predictors, "Dead", [])

    invert_predictors = []
    for i, feature_name in enumerate(predictors):
        logist_reg = LogisticRegression()
        logist_reg.fit(data.x[:, i].reshape(-1, 1), data.y)
        weight = logist_reg.coef_.ravel()[0]
        if weight < 0:
            invert_predictors.append(feature_name)

    return invert_predictors


def print_model(model, data):
    print("=" * 10 + "\nМодель")
    print("Первичные пороги:")
    for k, feature_name in enumerate(predictors):
        val = data.get_coord(feature_name, model.primary_thresholds[k])
        s = '≤' if feature_name in data.inverted_predictors else '≥'
        print(feature_name, " ", s, val, sep='')
    print("Фенотипы:")
    for j in range(len(predictors)):
        if len(model.phenotypes[j]) == 0:
            print(predictors[j])
        else:
            print(predictors[j], '& (', end='')
            for k in range(len(model.phenotypes[j])):
                if k > 0:
                    print(' | ', end='')
                feature_name = predictors[model.phenotypes[j][k]['feature']]
                val = data.get_coord(feature_name,model.phenotypes[j][k]['threshold'])
                print(feature_name, '[', val, ']', end='')
            print(')')
    print("Веса:", model.logistic_model.coef_.ravel())



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



if data_file == 'AF':
    data = Data("STEMI.xlsx", STEMI=True)
else:
    data = Data("DataSet.xlsx")
if data_file == 'AF':
    #predictors = ['Возраст', 'NER1', 'SIRI', 'СОЭ', 'TIMI после', 'СДЛА', 'Killip',
    #             'RR 600-1200', 'интервал PQ 120-200', 'EOS', 'NEUT']
    predictors = ['Возраст', 'NER1', 'SIRI', 'СОЭ', 'TIMI после', 'СДЛА', 'Killip',
                  'RR 600-1200', 'интервал PQ 120-200']
else:
    predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
invert_predictors = find_predictors_to_invert(data, predictors)
if data_file == 'AF':
    predictors.append('RR 600-1200_')
    invert_predictors.append('RR 600-1200_')
    predictors.append('интервал PQ 120-200_')
    invert_predictors.append('интервал PQ 120-200_')
print("Inverted:", invert_predictors)

if data_file == 'AF':
    data.prepare(predictors, "isAFAfter", invert_predictors)
else:
    data.prepare(predictors, "Dead", invert_predictors)

if data_file == 'AF':
    normal_thresholds = [0, 0, 0, 0, 2.5, 0, 0, 1200, 200, 600, 120]
    # Внимательно посмотреть, инвертируется ли RR! (зависит от датасета)
else:
    normal_thresholds = [0, 80, 3, 115, 50, 0, 1.0, 0.25, 5.6, 115]

min_thresholds = []
for i, nt in enumerate(normal_thresholds):
    val_normal = nt
    if predictors[i] in invert_predictors:
        val_normal = -val_normal
    val = (val_normal - data.scaler_mean[i]) / data.scaler_scale[i]
    min_thresholds.append(val)
# min_thresholds -- минимально допустимый порог (в преобразованных координатах)

if data_file == 'AF':
    threshold = 0.12
else:
    threshold = 0.03

num_splits = 1

if data_file == 'AF':
    random_state = 1234
else:
    random_state = 123

csvfile = open('splits.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
csvwriter.writerow(["auc_cv", "sen_cv", "spec_cv", "auc_test", "sen_test", "spec_test"])

for it in range(1, 1 + num_splits):
    np.random.seed(random_state + it - 1)
    print("SPLIT #", it, "of", num_splits)
    x_train, x_test, y_train, y_test = \
        train_test_split(data.x, data.y, test_size=0.2, stratify=data.y, random_state=random_state)  # закомментировать random_state

    model = MaxAUCPhenotypesModel()
    #auc_cv, sens_cv, spec_cv = model.fit(x_train, y_train)
    model.fit(x_train, y_train)
    print_model(model, data)
    auc_test, sen_test, spec_test = t_model(model, x_test, y_test, threshold)

