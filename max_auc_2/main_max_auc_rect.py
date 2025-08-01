import sys
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as sklearn_metrics
from sklearn.model_selection import train_test_split
import numpy as np
from sortedcontainers import SortedList
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm

data_file = 'AF'  # 'M'
#data_file = 'M'


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


def find_threshold_rect(x_, y_, labels_, min_x, min_y):
    # сортируем точки по убыванию y
    ind = np.argsort(y_)
    ind = ind[::-1]
    x = x_[ind]
    y = y_[ind]
    labels = labels_[ind]
    n = x.size
    N1 = np.sum(labels)  # число единиц
    N0 = n - N1  # число нулей

    max_auc = 0.0
    a = min_x  #None
    b = min_y  #None

    x_coord = SortedList()  # x-координаты всех добавленных точек
    ones_x_coord = SortedList()  # x-координаты всех добавленных единиц
    for i in range(n):
        x_coord.add(x[i])
        if labels[i] == 1:
            ones_x_coord.add(x[i])
            for j, x1 in enumerate(reversed(ones_x_coord)):
                # ones_right - число единиц не левее x1
                ones_right = j + 1
                # points_right - число точек не левее x1
                points_right = (i + 1) - x_coord.index(x1)
                # zeros_right - число нулей не левее x1
                zeros_right = points_right - ones_right

                auc = 0.5 + 0.5 * ones_right / N1 - 0.5 * zeros_right / N0

                if auc > max_auc and x1 >= min_x and y[i] >= min_y:
                    max_auc = auc
                    a = x1
                    b = y[i]

    return a, b, max_auc


def plot_2d(x1, x1_name, x1_plot_name, x2, x2_name, x2_plot_name, y, a, b, file_name=None):
    min_x1 = np.min(x1)
    min_x2 = np.min(x2)
    max_x1 = np.max(x1)
    max_x2 = np.max(x2)

    val_x1 = np.array([data.get_coord(x1_name, x1[i]) for i in range(x1.shape[0])])
    val_x2 = np.array([data.get_coord(x2_name, x2[i]) for i in range(x2.shape[0])])
    val_a = data.get_coord(x1_name, a)
    val_b = data.get_coord(x2_name, b)

    max_val_x1 = np.max(val_x1)
    max_val_x2 = np.max(val_x2)

    if x1_name == "NER1":
        max_val_x1 = 2000
        val_x2 = val_x2[val_x1 < max_val_x1]
        y = y[val_x1 < max_val_x1]
        val_x1 = val_x1[val_x1 < max_val_x1]
    if x2_name == "NER1":
        max_val_x2 = 2000
        val_x1 = val_x1[val_x2 < max_val_x2]
        y = y[val_x2 < max_val_x2]
        val_x2 = val_x2[val_x2 < max_val_x2]

    plt.scatter(val_x1[y == 0], val_x2[y == 0], marker='.', c='blue')  #, alpha=0.5)  #, linewidths=1)
    plt.scatter(val_x1[y == 1], val_x2[y == 1], marker='x', c='red')  #alpha=0.5
    plt.axline((val_a, val_b), (val_a, max_val_x2), c='green')
    plt.axline((val_a, val_b), (max_val_x1, val_b), c='green')
    plt.xlabel(x1_plot_name)
    plt.ylabel(x2_plot_name)
    if file_name is not None:
        plt.savefig(file_name, dpi=300)
    plt.show()


class MaxAUCRectModel:
    def __init__(self, num_combined_features):
        self.num_combined_features = num_combined_features
        self.features_used = None
        self.model = None
        self.thresholds = None

    def fit(self, x, y):
        data_size, num_features = x.shape[0], x.shape[1]
        z = None
        self.thresholds = []
        for ind1 in range(num_features):
            for ind2 in range(ind1 + 1, num_features):
                print("ind = ", ind1, "," , ind2)
                a, b, auc1 = find_threshold_rect(x[:, ind1], x[:, ind2], y[:])
                print(a, b)
                print("AUC =", auc1)
                row = np.zeros(data_size, dtype=int)
                for i in range(data_size):
                    if x[i, ind1] >= a and x[i, ind2] >= b:
                        row[i] = 1
                if z is None:
                    z = row
                else:
                    z = np.vstack((z, row))
                self.thresholds.append({'a': a, 'b': b, 'auc': auc1})

        z = z.T
        print(z)
        print(self.thresholds)

        # Отбор признаков методом включения
        max_auc = 0.0
        best_feature = None
        num_pair_features = z.shape[1]
        for feature in range(num_pair_features):
            print("Признак", feature + 1)
            model = LogisticRegression(solver='lbfgs', max_iter=10000)
            model.fit(z[:, feature].reshape(-1, 1), y)
            p = model.predict_proba(z[:, feature].reshape(-1, 1))[:, 1]
            auc = sklearn_metrics.roc_auc_score(y, p)
            if auc > max_auc:
                max_auc = auc
                best_feature = feature
        #print(best_feature, max_auc)

        features_used = [best_feature]

        for feature_cnt in range(1, self.num_combined_features):
            max_auc = 0.0
            best_feature = None
            for feature in range(num_pair_features):
                if feature not in features_used:
                    features_used.append(feature)
                    model = LogisticRegression(solver='lbfgs', max_iter=10000)
                    model.fit(z[:, features_used], y)
                    p = model.predict_proba(z[:, features_used])[:, 1]
                    auc = sklearn_metrics.roc_auc_score(y, p)
                    if auc > max_auc:
                        max_auc = auc
                        best_feature = feature
                    features_used.pop()
            features_used.append(best_feature)
            print("AUC =", max_auc)

        model = LogisticRegression(solver='lbfgs', max_iter=10000)
        model.fit(z[:, features_used], y)
        self.model = model
        self.features_used = features_used


    def fit_cross_val(self, x, y, x_final_test, y_final_test):
        data_size, num_features = x.shape[0], x.shape[1]
        NSPLITS = 5
        skf = StratifiedKFold(n_splits=NSPLITS)
        # Находим для каждой пары признаков средний порог и средний AUC
        z = None
        self.thresholds = []
        for ind1 in range(num_features):
            for ind2 in range(ind1 + 1, num_features):
                print("ind = ", ind1, "," , ind2)
                params = []
                for fold, (train_index, test_index) in enumerate(skf.split(x, y)):
                    x_train, x_test = x[train_index, :], x[test_index, :]
                    y_train, y_test = y[train_index], y[test_index]
                    print("  Fold", fold)
                    # находим пороги на обучающей выборке
                    a, b, auc_train = find_threshold_rect(x_train[:, ind1], x_train[:, ind2], y_train[:], min_thresholds[ind1], min_thresholds[ind2])
                    # учитываем минимально допустимые пороги
                    a = max(a, min_thresholds[ind1])
                    b = max(b, min_thresholds[ind2])
                    print(a, b)
                    print("  AUC на обучающей =", auc_train)
                    # считаем AUC на тестовой выборке
                    y_pred = np.zeros_like(y_test, dtype=int)
                    for i in range(x_test.shape[0]):
                        if x_test[i, ind1] >= a and x_test[i, ind2] >= b:
                            y_pred[i] = 1
                    auc_test = sklearn_metrics.roc_auc_score(y_test, y_pred)
                    print("  AUC на тестовой =", auc_test)
                    params.append({'a': a, 'b': b, 'auc': auc_test})
                mean_auc = np.mean([p['auc'] for p in params])
                std_auc = np.std([p['auc'] for p in params], ddof=1)
                print("Средний AUC =", mean_auc)
                # усредняем пороги
                mean_a = np.mean([p['a'] for p in params])
                mean_b = np.mean([p['b'] for p in params])
                # считаем AUC на итоговом тестировании
                y_pred = np.zeros_like(y_final_test, dtype=int)
                for i in range(x_final_test.shape[0]):
                    if x_final_test[i, ind1] >= mean_a and x_final_test[i, ind2] >= mean_b:
                        y_pred[i] = 1
                auc_final_test = sklearn_metrics.roc_auc_score(y_final_test, y_pred)
                print("AUC на итоговом тестировании =", auc_final_test)
                # оцениваем доверительный интервал для AUC на итоговом тестировании
                ntest = len(y_final_test)  # сколько раз разыгрываем случайный выбор
                mtest = 9 * (ntest // 10)  # сколько выбираем наблюдений
                auc_instances = []
                for _ in range(ntest):
                    indices = np.random.choice(len(y_final_test), mtest, replace=False)
                    auc_val = sklearn_metrics.roc_auc_score(y_final_test[indices], y_pred[indices])
                    auc_instances.append(auc_val)
                mean_auc_test = np.mean(auc_instances)
                std_auc_test = np.std(auc_instances, ddof=1)
                # сохраняем все параметры для данной пары предикторов
                self.thresholds.append({'a': mean_a, 'b': mean_b,
                                        'auc_train': mean_auc, 'auc_test': auc_final_test,
                                        'std_auc_train': std_auc, 'nsplits': NSPLITS,
                                        'ntest': ntest, 'mean_auc_test': mean_auc_test, 'std_auc_test': std_auc_test})
                # вычисляем дихотомизированный признак для данной пары предикторов
                row = np.zeros(data_size, dtype=int)
                for i in range(data_size):
                    if x[i, ind1] >= mean_a and x[i, ind2] >= mean_b:
                        row[i] = 1
                if z is None:
                    z = row
                else:
                    z = np.vstack((z, row))
        z = z.T
        print(z)
        print(self.thresholds)

        # Отбор признаков методом включения
        # сначала выбираем один признак с наибольшим AUC
        max_auc = 0.0
        best_feature = None
        num_pair_features = z.shape[1]
        for feature in range(num_pair_features):
            print("Признак", feature + 1)
            model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)
            model.fit(z[:, feature].reshape(-1, 1), y)
            p = model.predict_proba(z[:, feature].reshape(-1, 1))[:, 1]
            auc = sklearn_metrics.roc_auc_score(y, p)
            if auc > max_auc:
                max_auc = auc
                best_feature = feature
        #print(best_feature, max_auc)

        features_used = [best_feature]

        # затем добавляем каждый раз тот признак, который дает наибольший прирост AUC
        for feature_cnt in range(1, self.num_combined_features):
            max_auc = 0.0
            best_feature = None
            for feature in range(num_pair_features):
                if feature not in features_used:
                    features_used.append(feature)
                    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)
                    model.fit(z[:, features_used], y)
                    p = model.predict_proba(z[:, features_used])[:, 1]
                    auc = sklearn_metrics.roc_auc_score(y, p)
                    if auc > max_auc:
                        max_auc = auc
                        best_feature = feature
                    features_used.pop()
            features_used.append(best_feature)
            print("AUC =", max_auc)

        model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)
        model.fit(z[:, features_used], y)
        self.model = model
        self.features_used = features_used

        z1 = z[:, features_used]
        z1 = sm.add_constant(z1)
        sm_model = sm.Logit(y, z1)
        result = sm_model.fit_regularized()
        print(result.summary())


    def predict_proba(self, x):
        data_size, num_features = x.shape[0], x.shape[1]
        z = None
        k = 0
        for ind1 in range(num_features):
            for ind2 in range(ind1 + 1, num_features):
                row = np.zeros(data_size, dtype=int)
                a = self.thresholds[k]['a']
                b = self.thresholds[k]['b']
                for i in range(data_size):
                    if x[i, ind1] >= a and x[i, ind2] >= b:
                        row[i] = 1
                if z is None:
                    z = row
                else:
                    z = np.vstack((z, row))
                k += 1

        z = z.T

        return self.model.predict_proba(z[:, self.features_used])

if data_file == 'AF':
    data = Data("STEMI.xlsx", STEMI=True)
else:
    data = Data("DataSet.xlsx")
if data_file == 'AF':
    predictors = ['Возраст', 'NER1', 'SIRI', 'СОЭ', 'TIMI после', 'СДЛА', 'Killip',
                  'RR 600-1200', 'интервал PQ 120-200']
    predictors_eng = predictors
    predictors_rus = predictors
else:
    predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
    predictors_eng = ["Age, years", "HR, bpm", "Killip class", "Cr, umol/l", "EF LV, %", "NEUT, %", "EOS, %", "PCT, %", "Glu, mmol/l", "SBP, mmHg"]
    predictors_rus = ["Возраст, лет", "ЧСС в минуту", "Класс ОСН по T. Killip", "Креатинин, мкмоль/л", "Фракция выброса левого желудочка, %", "Нейтрофилы, %", "Эозинофилы, %", "Тромбокрит, %", "Глюкоза, ммоль/л", "Систолическое АД, мм рт.ст."]
invert_predictors = find_predictors_to_invert(data, predictors)
if data_file == 'AF':
    predictors.append('RR 600-1200_')
    #invert_predictors.append('RR 600-1200_')
print("Inverted", invert_predictors)
if data_file == 'AF':
    data.prepare(predictors, "isAFAfter", invert_predictors)
else:
    data.prepare(predictors, "Dead", invert_predictors)

if data_file == 'AF':
    #normal_thresholds = [0, 0, 0, 0, 2, 0, 0, 0, 500, 2000]
    normal_thresholds = [0, 0, 0, 0, 2, 0, 0, 600, 200, 1200]
else:
    normal_thresholds = [0, 80, 3, 115, 50, 0, 100, 0.25, 5.6, 115]
#normal_thresholds = [0, 0, 0, 0, 1000, 0, 1000, 0, 0, 1000]
min_thresholds = []
for i, nt in enumerate(normal_thresholds):
    val_normal = nt
    if predictors[i] in invert_predictors:
        val_normal = -val_normal
    val = (val_normal - data.scaler_mean[i]) / data.scaler_scale[i]
    min_thresholds.append(val)
# min_thresholds -- минимально допустимый порог (в преобразованных координатах)


"""
ind1 = 0
ind2 = 5

a, b, auc = find_threshold_rect(data.x[:, ind1], data.x[:, ind2], data.y[:])
print(a, b)
print("AUC =", auc)
plot_2d(data.x[:, ind1], predictors[ind1], data.x[:, ind2], predictors[ind2], data.y[:], a, b)
"""

"""
ind1 = 0
ind2 = 6
plot_2d(data.x[:, ind1], predictors[ind1], predictors_eng[ind1],
        data.x[:, ind2], predictors[ind2], predictors_eng[ind2], data.y[:],
        data.invert_coord(predictors[ind1], 51), data.invert_coord(predictors[ind2], 0.54),
        file_name="fig_rect/" + predictors[ind1] + "_" + predictors[ind2] + "_eng.png")
"""


if data_file == 'AF':
    threshold = 0.12
else:
    threshold = 0.03  #0.04

num_combined_features = 12  #10

num_splits = 1

if data_file == 'AF':
    random_state = 1234
else:
    random_state = 123

csvfile = open('splits.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
csvwriter.writerow(["auc", "sen", "spec"])

for it in range(1, 1 + num_splits):
    print("SPLIT #", it, "of", num_splits)

    x_train, x_test, y_train, y_test = \
        train_test_split(data.x, data.y, test_size=0.2, stratify=data.y, random_state=random_state)  # закомментировать random_state

    max_auc_rect_model = MaxAUCRectModel(num_combined_features)
    #max_auc_rect_model.fit(x_train, y_train)
    max_auc_rect_model.fit_cross_val(x_train, y_train, x_test, y_test)

    print("Обучена модель")
    data_size, num_features = data.x.shape[0], data.x.shape[1]
    #print(max_auc_2_model.model.coef_.ravel())

    k = 0
    for ind1 in range(num_features):
        for ind2 in range(ind1 + 1, num_features):
            if k in max_auc_rect_model.features_used:
                print(ind1, ind2, max_auc_rect_model.model.coef_[0][max_auc_rect_model.features_used.index(k)])
                print(max_auc_rect_model.thresholds[k])
                a = max_auc_rect_model.thresholds[k]['a']
                b = max_auc_rect_model.thresholds[k]['b']

                feature1 = predictors[ind1]
                feature2 = predictors[ind2]
                val_a = data.get_coord(feature1, a)
                val_b = data.get_coord(feature2, b)
                s1 = '≤' if feature1 in data.inverted_predictors else '≥'
                s2 = '≤' if feature2 in data.inverted_predictors else '≥'
                print('  ', feature1, " ", s1, val_a, sep='')
                print('  ', feature2, " ", s2, val_b, sep='')
                #print("  AUC =", max_auc_rect_model.thresholds[k]['auc'])
                print("  AUC (обучающая) =", max_auc_rect_model.thresholds[k]['auc_train'])
                print("  Оценка: ", "[",
                      max_auc_rect_model.thresholds[k]['auc_train']
                      - 1.96 * max_auc_rect_model.thresholds[k]['std_auc_train'] / np.sqrt(max_auc_rect_model.thresholds[k]['nsplits']),
                      ",",
                      max_auc_rect_model.thresholds[k]['auc_train']
                      + 1.96 * max_auc_rect_model.thresholds[k]['std_auc_train'] / np.sqrt(max_auc_rect_model.thresholds[k]['nsplits']),
                      "]")
                print("  AUC (тестовая) =", max_auc_rect_model.thresholds[k]['auc_test'])
                print("  Оценка: ", max_auc_rect_model.thresholds[k]['mean_auc_test'], "[",
                      max_auc_rect_model.thresholds[k]['mean_auc_test']
                      - 1.96 * max_auc_rect_model.thresholds[k]['std_auc_test'] / np.sqrt(max_auc_rect_model.thresholds[k]['ntest']),
                      ",",
                      max_auc_rect_model.thresholds[k]['mean_auc_test']
                      + 1.96 * max_auc_rect_model.thresholds[k]['std_auc_test'] / np.sqrt(max_auc_rect_model.thresholds[k]['ntest']),
                      "]")
                plot_2d(data.x[:, ind1], predictors[ind1], predictors_eng[ind1],
                        data.x[:, ind2], predictors[ind2], predictors_eng[ind2], data.y[:], a, b,
                        file_name="fig_rect/" + predictors[ind1] + "_" + predictors[ind2] + ".png")
                plot_2d(data.x[:, ind1], predictors[ind1], predictors_rus[ind1],
                        data.x[:, ind2], predictors[ind2], predictors_rus[ind2], data.y[:], a, b,
                        file_name="fig_rect/" + predictors[ind1] + "_" + predictors[ind2] + "_rus.png")
            k += 1


    p = max_auc_rect_model.predict_proba(x_test)[:, 1]
    auc = sklearn_metrics.roc_auc_score(y_test, p)
    print("AUC:", auc)
    # выводим качество модели
    y_pred = np.where(p > threshold, 1, 0)
    tn, fp, fn, tp = sklearn_metrics.confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print("Sens:", sensitivity, "Spec:", specificity)
    print("tp =", tp, "fn =", fn, "fp =", fp, "tn =", tn)

    csvwriter.writerow(map(str, [auc, sensitivity, specificity]))
