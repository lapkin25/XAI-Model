import numpy as np
import pandas as pd
from woodtapper import WoodTapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(1, '../dichotomization')

from read_data import Data

data = Data("DataSet.xlsx")

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]

data.prepare(predictors, "Dead", [], scale_data=False)

data_np = np.column_stack([data.x, data.y])

#print(data_np)

columns = predictors + ['Dead']
df = pd.DataFrame(data_np, columns=columns)

X = df[predictors]
y = df['Dead']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучаем Random Forest (аналог SIRUS-леса)
rf = RandomForestClassifier(
    n_estimators=100,        # количество деревьев
    max_depth=5,             # ограничиваем глубину для простоты правил
    min_samples_leaf=10,
    random_state=42
)
rf.fit(X_train, y_train)

print(f"Точность Random Forest на тесте: {rf.score(X_test, y_test):.3f}")

# ------------------ 3. Извлечение интерпретируемых правил с помощью WoodTapper ------------------
# Инициализируем WoodTapper для извлечения правил из обученного леса
wt = WoodTapper(rf, feature_names=X.columns.tolist())

# Извлекаем все правила из деревьев
rules = wt.get_rules()

# Преобразуем правила в DataFrame для анализа
rules_df = pd.DataFrame(rules)

# Добавляем метрики качества для каждого правила (как в subgroup discovery)
# Вычисляем поддержку (support) и точность (precision) каждого правила на обучающей выборке
def evaluate_rule(rule, X_data, y_data):
    """Оценка качества правила: поддержка и точность"""
    mask = rule.evaluate(X_data)  # метод evaluate должен возвращать boolean-маску
    support = mask.sum()
    if support == 0:
        return 0.0, 0.0
    precision = (y_data[mask] == 1).sum() / support
    return support / len(X_data), precision

# Добавляем колонки с метриками (адаптируйте под структуру ваших правил)
# В WoodTapper правила обычно имеют метод .evaluate() или .predict()
support_list = []
precision_list = []
for rule in rules:
    # Если правило — это объект с методом evaluate
    if hasattr(rule, 'evaluate'):
        mask = rule.evaluate(X_train.values)
    else:
        # Если правило — это строка с условием, используйте query
        mask = X_train.eval(rule) if isinstance(rule, str) else np.ones(len(X_train), dtype=bool)

    support = mask.sum()
    if support > 0:
        precision = (y_train[mask] == 1).sum() / support
    else:
        precision = 0.0
    support_list.append(support / len(X_train))
    precision_list.append(precision)

rules_df['support'] = support_list
rules_df['precision'] = precision_list

# Сортируем правила по качеству (например, по точности при поддержке > 5%)
top_rules = rules_df[rules_df['support'] > 0.05].sort_values('precision', ascending=False)

print("\n=== Топ-10 правил (аналог подгрупп) ===")
print(top_rules[['support', 'precision', 'rule']].head(10))

# ------------------ 4. Визуализация важности правил (опционально) ------------------
# WoodTapper позволяет визуализировать деревья и правила
# wt.visualize_tree(tree_id=0)  # визуализация конкретного дерева

# ------------------ 5. Создание SIRUS-подобного классификатора из правил ------------------
# Отбираем топ-K правил для построения интерпретируемого классификатора
K = 10
selected_rules = top_rules.head(K)['rule'].tolist()

# Функция предсказания на основе правил (голосование)
def predict_with_rules(X_data, rules_list):
    predictions = np.zeros(len(X_data))
    for rule in rules_list:
        if hasattr(rule, 'evaluate'):
            mask = rule.evaluate(X_data.values)
        else:
            mask = X_data.eval(rule) if isinstance(rule, str) else np.ones(len(X_data), dtype=bool)
        predictions[mask] += 1
    # Если правило сработало хотя бы раз — предсказываем 1
    return (predictions > 0).astype(int)

# Оцениваем качество интерпретируемого классификатора
y_pred_rules = predict_with_rules(X_test, selected_rules)
accuracy_rules = (y_pred_rules == y_test).mean()
print(f"\nТочность классификатора на основе {K} правил: {accuracy_rules:.3f}")
