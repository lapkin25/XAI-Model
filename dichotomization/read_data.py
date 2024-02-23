# Источник данных: https://github.com/NikitaKuksin/PublicCodAticle

import pandas as pd
from sklearn import preprocessing


class Data:
    def __init__(self, dataset_path):
        self.dataset = pd.read_excel(dataset_path)
        self.predictors = []
        self.output = ""
        self.inverted_predictors = []
        self.x = None
        self.y = None
        self.scaler_mean = None
        self.scaler_scale = None

    def prepare(self, selected_predictors, output_feature, invert_predictors):
        """
        selected_predictors - список названий признаков
        output_feature - название выходного признака
        invert_predictors - список названий признаков с отрицательными весами
        """
        self.predictors = selected_predictors
        self.output = output_feature
        self.inverted_predictors = invert_predictors

        data_x_y = self.dataset[selected_predictors + [output_feature]].dropna()
        for feature_name in invert_predictors:
            data_x_y[feature_name] = -data_x_y[feature_name]
        data_x = data_x_y[selected_predictors].to_numpy()
        data_y = data_x_y[output_feature].to_numpy()

        scaler = preprocessing.StandardScaler().fit(data_x)
        self.scaler_mean = scaler.mean_
        self.scaler_scale = scaler.scale_
        self.x = scaler.transform(data_x)
        self.y = data_y

    def get_coord(self, feature_name, scaled_val):
        """
        Переход к первоначальным координатам
        """
        feature = self.predictors.index(feature_name)
        val = self.scaler_mean[feature] + self.scaler_scale[feature] * scaled_val
        if feature_name in self.inverted_predictors:
            val = -val
        return val
