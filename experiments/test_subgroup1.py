import numpy as np
import pandas as pd
import pysubgroup as ps

import sys
sys.path.insert(1, '../dichotomization')

from dichotomization.read_data import Data

data = Data("DataSet.xlsx")

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]

data.prepare(predictors, "Dead", [], scale_data=False)

data_np = np.column_stack([data.x, data.y])

#print(data_np)

columns = predictors + ['Dead']
df = pd.DataFrame(data_np, columns=columns)


# ------------------ 3. Определение целевой переменной ------------------
# Целевая бинарная (можно использовать и числовую, но для примера - бинарная)
target = ps.BinaryTarget('Dead', True)   # ищем подгруппы, где target == 1

searchspace = ps.create_selectors(df, ignore=['Dead'])
task = ps.SubgroupDiscoveryTask (
    df,
    target,
    searchspace,
    result_set_size=15,
    depth=2,
    qf=ps.WRAccQF())
result = ps.DFS().execute(task)

# Чтобы показать все столбцы
pd.set_option('display.max_columns', None)
print(result.to_dataframe())

#print(len(df[df['NEUT'] >= 77.3]))
#print(len(df[(df['NEUT'] >= 77.3) & (df['Dead'] == 1)]))