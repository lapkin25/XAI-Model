# Источник данных: https://github.com/NikitaKuksin/PublicCodAticle

import numpy as np
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

    def binarize(self, selected_predictors, output_feature, invert_predictors, thresholds):
        data_x_y = self.dataset[selected_predictors + [output_feature]].dropna()
        data_x = data_x_y[selected_predictors].to_numpy()
        data_y = data_x_y[output_feature].to_numpy()
        bin_x = np.empty_like(data_x)
        for i, pred in enumerate(selected_predictors):
            if pred in invert_predictors:
                bin_x[:, i] = np.where(data_x[:, i] <= thresholds[i], 1, 0)
            else:
                bin_x[:, i] = np.where(data_x[:, i] >= thresholds[i], 1, 0)
        return bin_x

    def binarize_combined(self, selected_predictors, output_feature, invert_predictors, thresholds,
                          combined_predictors, combined_thresholds):
        data_x_y = self.dataset[selected_predictors + [output_feature]].dropna()
        data_x = data_x_y[selected_predictors].to_numpy()
        data_y = data_x_y[output_feature].to_numpy()
        data_size = data_x.shape[0]
        bin_x = np.zeros((data_size, len(combined_predictors)))
        for i, (pred1, pred2) in enumerate(combined_predictors):
            j = selected_predictors.index(pred1)
            k = selected_predictors.index(pred2)
            for s in range(data_size):
                if pred1 in invert_predictors:
                    bin_j = (data_x[s, j] <= thresholds[j])
                else:
                    bin_j = (data_x[s, j] >= thresholds[j])
                if pred2 in invert_predictors:
                    bin_k = (data_x[s, k] <= combined_thresholds[i])
                else:
                    bin_k = (data_x[s, k] >= combined_thresholds[i])
                if bin_j and bin_k:
                    bin_x[s, i] = 1
                else:
                    bin_x[s, i] = 0
        return bin_x
