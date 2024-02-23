from read_data import Data

def find_predictors_to_invert(data, predictors):
    # TODO: обучить логистическую регрессию с выделенными признаками,
    #   выбрать признаки с отрицательными весами
    pass


data = Data("DataSet.xlsx")
predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
invert_predictors = []
data.prepare(predictors, "Dead", invert_predictors)
#print(data.x)
#print(data.y)
