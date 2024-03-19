import numpy as np
from read_data import Data
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as sklearn_metrics
from sklearn.model_selection import StratifiedKFold, train_test_split


def test_model(model, data_x, data_y, p_threshold=0.05):
    print("Кросс-валидация")
    skf = StratifiedKFold(n_splits=7)
    for i, (train_index, test_index) in enumerate(skf.split(data_x, data_y)):
        print(f"Fold {i}:")
        x_train = data_x[train_index, :]
        y_train = data_y[train_index]
        x_test = data_x[test_index, :]
        y_test = data_y[test_index]
        model.fit(x_train, y_train)
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

    print("Итоговое тестирование")
    x_train, x_test, y_train, y_test = \
        train_test_split(data_x, data_y, test_size=0.2, random_state=123, stratify=data_y)
    model.fit(x_train, y_train)
    p = model.predict_proba(x_test)[:, 1]
    auc = sklearn_metrics.roc_auc_score(y_test, p)
    print("AUC:", auc)
    # выводим качество модели
    y_pred = np.where(p > p_threshold, 1, 0)
    tn, fp, fn, tp = sklearn_metrics.confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print("Веса:", model.coef_.ravel())
    print("Sens:", sensitivity, "Spec:", specificity)
    print("tp =", tp, "fn =", fn, "fp =", fp, "tn =", tn)


data = Data("DataSet.xlsx")
predictors = ["Cr", "HR", "EF LV", "Glu", "NEUT", "EOS", "PCT", "SBP", "Age", "Killip class"]
thresholds = [229.44, 60.0, 30.0, 6.19, 89.2, 0.5, 0.32, 115.0, 68.0, 4.0]
#invert_predictors = [False, False, True, False, False, True, False, True, False, False]
invert_predictors = ["EF LV", "EOS", "SBP"]

combined_predictors = [("Age", "EOS"), ("HR", "Age"), ("HR", "Glu"), ("Killip class", "NEUT"),
                       ("Glu", "EF LV"), ("Glu", "Age"), ("Age", "NEUT"), ("SBP", "EOS"),
                       ("Glu", "SBP"), ("Killip class", "HR"), ("EF LV", "Cr"),
                       ("Killip class", "EF LV"), ("NEUT", "EF LV"), ("SBP", "HR"), ("EOS", "Age")]
combined_thresholds = [0.0, 76.0, 8.98, 75.0, 40.0, 68.0, 76.6, 0.0, 90.0, 90.0, 105.61, 46.0, 54.62, 90.0, 64.0]

bin_x = data.binarize(predictors, "Dead", thresholds)
bin_x_combined = data.binarize_combined(predictors, "Dead", thresholds,
                                        combined_predictors, combined_thresholds)

data.prepare(predictors, "Dead", invert_predictors)

# модель с непрерывными признаками
print("Модель 1")
model1 = LogisticRegression()
test_model(model1, data.x, data.y)

# модель с бинаризованными индивидуальными признаками
print("Модель 2")
model2 = LogisticRegression()
test_model(model2, bin_x, data.y)

# модель с бинаризованными комбинированными признаками
print("Модель 3")
model3 = LogisticRegression()
test_model(model3, bin_x_combined, data.y)
