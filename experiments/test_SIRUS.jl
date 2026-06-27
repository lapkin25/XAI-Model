using Pkg
Pkg.add("XLSX")
Pkg.add("SIRUS")
Pkg.add("MLJ") # Для унифицированного интерфейса машинного обучения
Pkg.add("DataFrames") # Для удобной работы с табличными данными

# ============================
# 1. Загрузка библиотек
# ============================
using XLSX, DataFrames, MLJ, SIRUS

# ============================
# 2. Чтение данных из XLS-файла
# ============================
file_path = "/Dataset STEMI IHM.xlsx"

# Открываем файл и читаем первый лист в DataFrame
data = XLSX.readtable(file_path, "Лист1") |> DataFrame

# Проверим, что загрузилось
println("Размер данных: ", size(data))
println(first(data, 5))

data = dropmissing(data)

println("Размер данных: ", size(data))

# ============================
# 3. Подготовка данных для MLJ
# ============================
# Предположим, что целевая переменная — последний столбец
X = data[:, 1:end-1]  # Признаки
y = data[:, end]      # Целевая переменная



#println(X)
#println(y)
# Для классификации необходимо преобразовать y в категориальный массив
# Убедитесь, что ваша целевая переменная — это категории
# Пример: y = categorical(y)
y = categorical(y)
# Для регрессии y должен быть числовым вектором
# Пример: y = float.(y)

# ============================
# 4. Настройка и обучение модели SIRUS
# ============================
# Выберите одну из моделей в зависимости от задачи:
# - StableRulesClassifier для классификации [citation:1]
# - StableRulesRegressor для регрессии [citation:1]

model = StableRulesClassifier()  # или StableRulesRegressor()

# Создаем "машину" для обучения
mach = machine(model, X, y)

# Обучаем модель
fit!(mach)

# ============================
# 5. Извлечение и просмотр правил
# ============================
# Получаем обученную модель из машины
fitted_model = mach.fitresult

# Выводим полученные правила
println("Обученная модель SIRUS:")
println(fitted_model)  # Здесь будут перечислены все правила [citation:1][citation:12]

# ============================
# 6. (Опционально) Предсказание и оценка качества
# ============================
# Предсказание на тех же данных (для примера)
predictions = predict(mach, X)

#print(predictions)

# Оценка качества (для классификации)
if model isa StableRulesClassifier
    accuracy = MLJ.accuracy(predictions, y)
    println("Точность (Accuracy): ", accuracy)
else
    # Для регрессии можно использовать среднеквадратичную ошибку
    mse = MLJ.mse(predictions, y)
    println("Среднеквадратичная ошибка (MSE): ", mse)
end