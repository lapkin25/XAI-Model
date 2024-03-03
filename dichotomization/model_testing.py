from sklearn import metrics as sklearn_metrics
from sklearn.model_selection import StratifiedKFold, train_test_split


def test_model(model, data):
    print("Кросс-валидация")
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=123)
    for i, (train_index, test_index) in enumerate(skf.split(data.x, data.y)):
        print(f"Fold {i}:")
        x_train = data.x[train_index, :]
        y_train = data.y[train_index]
        x_test = data.x[test_index, :]
        y_test = data.y[test_index]
        model.fit(x_train, y_train)
        p = model.predict_proba(x_test, y_test)
        auc = sklearn_metrics.roc_auc_score(y_test, p)
        print("AUC:", auc)

    print("Итоговое тестирование")
    x_train, x_test, y_train, y_test = \
        train_test_split(data.x, data.y, test_size=0.2, random_state=123, stratify=data.y)
    model.fit(x_train, y_train)
    p = model.predict_proba(x_test, y_test)
    auc = sklearn_metrics.roc_auc_score(y_test, p)
    print("AUC:", auc)
