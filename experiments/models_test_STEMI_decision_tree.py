import numpy as np
import pandas as pd
import pysubgroup as ps

import sys
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data

data = Data("STEMI.xlsx", STEMI=True)

predictors = ['Возраст', 'NER1', 'SIRI', 'СОЭ', 'TIMI после', 'СДЛА', 'Killip',
              'RR 600-1200', 'интервал PQ 120-200']

data.prepare(predictors, "isAFAfter", [], scale_data=False)

data_np = np.column_stack([data.x, data.y])

#print(data_np)

columns = predictors + ['isAFAfter']
df = pd.DataFrame(data_np, columns=columns)


# =============================================
# 1. Установка пакетов (если ещё не установлены)
# =============================================
# pip install pandas openpyxl scikit-learn matplotlib

# =============================================
# 2. Подключение библиотек
# =============================================

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import matplotlib.pyplot as plt

# =============================================
# 3. Чтение данных из Excel-файла
# =============================================

# Читаем данные
data = df

print(f"Исходный размер данных: {data.shape}")

# =============================================
# 4. Удаление строк с пропусками (dropna)
# =============================================
data_clean = data.dropna()
print(f"Размер после очистки: {data_clean.shape}")
print(f"Удалено строк: {data.shape[0] - data_clean.shape[0]}")

if data_clean.empty:
    raise ValueError("Все строки содержат пропуски – обучение невозможно.")

# =============================================
# 5. Подготовка признаков (X) и целевой переменной (y)
# =============================================
# Предполагаем, что последний столбец – целевая переменная
X = data_clean.iloc[:, :-1]   # все столбцы, кроме последнего
y = data_clean.iloc[:, -1]    # последний столбец

# Для классификации убедимся, что y – категориальный
# (если это строки или числа-категории, можно оставить как есть)
# Для регрессии y должен быть числовым

# =============================================
# 6. Разделение на обучающую и тестовую выборки
# =============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2025
)

X_train = X
y_train = y
X_test = X
y_test = y

# =============================================
# 7. НАСТРОЙКА ГИПЕРПАРАМЕТРОВ И ОБУЧЕНИЕ МОДЕЛИ
# =============================================

# 7.1 Создаём модель с выбранными гиперпараметрами
# Для классификации:
model = DecisionTreeClassifier(
    max_depth=5,                # глубина дерева (аналог max_depth в SIRUS)
    min_samples_split=5,        # минимальное число образцов для разделения узла
    min_samples_leaf=5,         # минимальное число образцов в листе
    max_features="sqrt",        # число признаков для поиска лучшего разделения
    random_state=2025           # для воспроизводимости
)

# Для регрессии раскомментируйте следующую строку:
# model = DecisionTreeRegressor(
#     max_depth=3,
#     min_samples_split=5,
#     min_samples_leaf=5,
#     max_features="sqrt",
#     random_state=2025
# )

# 7.2 Обучаем модель
model.fit(X_train, y_train)

# =============================================
# 8. Просмотр структуры дерева (правила)
# =============================================
print("Структура дерева решений:")
print(model.tree_)  # сырая структура

# Более удобный вывод правил (текстовое представление)
from sklearn.tree import export_text
rules_text = export_text(model, feature_names=list(X.columns))
print("Правила дерева:")
print(rules_text)

# =============================================
# 9. Оценка качества на тестовой выборке
# =============================================
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]   # вероятность положительного класса

if isinstance(model, DecisionTreeClassifier):
    acc = accuracy_score(y_test, y_pred)
    print(f"Точность (Accuracy) на тесте: {acc:.4f}")
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"AUC (ROC-AUC) на тесте: {auc:.4f}")
else:
    mse = mean_squared_error(y_test, y_pred)
    print(f"Среднеквадратичная ошибка (MSE) на тесте: {mse:.4f}")

# =============================================
# 10. Визуализация дерева (опционально)
# =============================================
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=list(X.columns),
          class_names=[str(c) for c in sorted(y.unique())],
          filled=True, rounded=True)
plt.title("Дерево решений")
plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
plt.show()