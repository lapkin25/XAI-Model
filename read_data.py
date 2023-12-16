# Источник данных: https://github.com/NikitaKuksin/PublicCodAticle

import pandas as pd

# Принимает:
#   predictors - список имен входных признаков
#   output - имя выходного признака
# Возвращает:
#   x - двумерный массив значений входных признаков
#   y - одномерный массив значений выходного признака
def read_data(predictors, output):
    path_dataset = r'DataSet.xlsx'
    dataset = pd.read_excel(path_dataset)
    dataset = dataset[predictors + [output]].dropna()
    x = dataset[predictors].to_numpy()
    y = dataset[output].to_numpy()
    return x, y

