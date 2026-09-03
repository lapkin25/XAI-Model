import numpy as np

def compute_r_and_mape(file_path, delimiter=' '):
    """
    Читает из файла два столбца (разделитель по умолчанию — табуляция),
    считает коэффициент корреляции Пирсона (r) и MAPE.

    Параметры:
        file_path: путь к файлу
        delimiter: разделитель столбцов (по умолчанию '\t' для табуляции)

    Возвращает:
        r: коэффициент корреляции
        mape: средняя абсолютная процентная ошибка в процентах
    """
    # Загружаем данные: пропускаем пустые строки, читаем как float
    data = np.loadtxt(file_path, delimiter=delimiter)

    if data.ndim == 1:
        raise ValueError("Файл должен содержать ровно два столбца.")
    if data.shape[1] != 2:
        raise ValueError(f"Ожидается 2 столбца, а найдено {data.shape[1]}.")

    x = data[:, 0]  # фактические значения
    y = data[:, 1]  # прогнозные значения

    # Коэффициент корреляции Пирсона
    r = np.corrcoef(x, y)[0, 1]

    # MAPE (в процентах), с защитой от деления на ноль
    mask = x != 0
    if not np.any(mask):
        raise ValueError("Все значения в первом столбце равны нулю — MAPE не определён.")
    mape = np.mean(np.abs((x[mask] - y[mask]) / x[mask])) * 100

    return r, mape

if __name__ == "__main__":
    file_path = "data.txt"  # укажите имя вашего файла
    try:
        r, mape = compute_r_and_mape(file_path)
        print(f"Коэффициент корреляции (r): {r:.2f}")
        print(f"MAPE: {mape:.1f}%")
    except Exception as e:
        print(f"Ошибка: {e}")