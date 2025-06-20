# Источник данных: https://github.com/NikitaKuksin/PublicCodAticle

import numpy as np
import pandas as pd
from sklearn import preprocessing


class Data:
    def __init__(self, dataset_path, STEMI=False):
        self.dataset = pd.read_excel(dataset_path)
        self.predictors = []
        self.output = ""
        self.inverted_predictors = []
        self.x = None
        self.y = None
        self.scaler_mean = None
        self.scaler_scale = None
        if STEMI:
            # убрали неопределенность по ФП в анамнезе
            self.dataset = self.dataset.loc[self.dataset['ФП a (в анамнезе)'].isna() == False, :]
            self.dataset = self.dataset.loc[(self.dataset['ФП a (в анамнезе)'] == 'нет') &
                                            (self.dataset['ФП b (после чкв)'].isna() == False) &
                                            (self.dataset['ФП при окс (до чкв)'] == 'нет'), :]
            self.dataset.loc[(self.dataset['Эозинофилы (абсолютное значение)'] != 0.0) &
                             (self.dataset['Эозинофилы (абсолютное значение)'].isna() == False) &
                             (self.dataset['Нейтрофилы (абсолютное значение)(a)'].isna() == False), ('NER1')] =\
                ((self.dataset['Нейтрофилы (абсолютное значение)(a)']).astype(float)
                 / self.dataset['Эозинофилы (абсолютное значение)'].astype(float))
            self.dataset.loc[(self.dataset['Лимфоциты (абсолютное значение)'] != 0.0) &
                             (self.dataset['Лимфоциты (абсолютное значение)'].isna() == False) &
                             (self.dataset['Моноциты (абсолютное значение)'].isna() == False), ('MLR1')] =\
                ((self.dataset['Моноциты (абсолютное значение)']).astype(float)
                 / self.dataset['Лимфоциты (абсолютное значение)'].astype(float))
            name = 'SIRI'
            self.dataset.loc[(self.dataset['Лимфоциты (абсолютное значение)'] != 0.0) &
                             (self.dataset['Лимфоциты (абсолютное значение)'].isna() == False) &
                             (self.dataset['Моноциты (абсолютное значение)'].isna() == False), (name)] =(
                    (self.dataset['Нейтрофилы (абсолютное значение)(a)'].astype(float)
                     * self.dataset['Моноциты (абсолютное значение)']).astype(float)
                    / self.dataset['Лимфоциты (абсолютное значение)'].astype(float))
            name = 'NBR1'
            self.dataset.loc[:, (name)] = None
            self.dataset.loc[(self.dataset['Лимфоциты (абсолютное значение)'] != 0.0) &
                             (self.dataset['Лимфоциты (абсолютное значение)'].isna() == False) &
                             (self.dataset['Базофилы (абсолютное значение)'].isna() == False), (name)] =(
                    (self.dataset['Нейтрофилы (абсолютное значение)(a)'].astype(float)
                     * self.dataset['Моноциты (абсолютное значение)']).astype(float)
                    / self.dataset['Лимфоциты (абсолютное значение)'].astype(float))
            self.dataset.loc[:, ('isAFAfter')] = None
            self.dataset.loc[(self.dataset['ФП b (после чкв)'] == 'да'), ('isAFAfter')] = 1
            self.dataset.loc[(self.dataset['ФП b (после чкв)'] == 'нет'), ('isAFAfter')] = 0
            name = 'NLR'
            name1 = 'Нейтрофилы (абсолютное значение)(a)'
            name2 = 'Лимфоциты (абсолютное значение)'
            #self.dataset[name] = np.NaN
            self.dataset[name] = self.dataset[(name1)] / self.dataset[(name2)]
            name = 'SII'
            name1 = 'Тромбоциты'
            name2 = 'NLR'
            #self.dataset[name] = np.NaN
            self.dataset[name] = self.dataset[(name1)] * self.dataset[(name2)]


    def prepare(self, selected_predictors, output_feature, invert_predictors, scale_data=True):
        """
        selected_predictors - список названий признаков
        output_feature - название выходного признака
        invert_predictors - список названий признаков с отрицательными весами
        """
        self.predictors = selected_predictors
        self.output = output_feature
        self.inverted_predictors = invert_predictors

        print("До исключения пропусков:", self.dataset[selected_predictors + [output_feature]].shape, np.sum(self.dataset[output_feature].dropna().to_numpy(dtype=int)))

        data_x_y = self.dataset[selected_predictors + [output_feature]].dropna()

        print("После исключения пропусков:", data_x_y.shape)

        for feature_name in invert_predictors:
            data_x_y[feature_name] = -data_x_y[feature_name]
        data_x = data_x_y[selected_predictors].to_numpy()
        data_y = data_x_y[output_feature].to_numpy(dtype=int)

        print("Умерших", np.sum(data_y))

        if scale_data:
            scaler = preprocessing.StandardScaler().fit(data_x)
            self.scaler_mean = scaler.mean_
            self.scaler_scale = scaler.scale_
            self.x = scaler.transform(data_x)
            self.y = data_y
        else:
            self.x = data_x
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

    def invert_coord(self, feature_name, val_):
        feature = self.predictors.index(feature_name)
        val = val_
        if feature_name in self.inverted_predictors:
            val = -val
        scaled_val = (val - self.scaler_mean[feature]) / self.scaler_scale[feature]
        return scaled_val

    def binarize(self, selected_predictors, output_feature, thresholds):
        data_x_y = self.dataset[selected_predictors + [output_feature]].dropna()
        data_x = data_x_y[selected_predictors].to_numpy()
        data_y = data_x_y[output_feature].to_numpy()
        bin_x = np.empty_like(data_x)
        for i, pred in enumerate(selected_predictors):
            if pred in self.inverted_predictors:
                bin_x[:, i] = np.where(data_x[:, i] <= thresholds[i], 1, 0)
            else:
                bin_x[:, i] = np.where(data_x[:, i] >= thresholds[i], 1, 0)
        return bin_x

    def binarize_combined(self, selected_predictors, output_feature, thresholds,
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
                if pred1 in self.inverted_predictors:
                    bin_j = (data_x[s, j] <= thresholds[j])
                else:
                    bin_j = (data_x[s, j] >= thresholds[j])
                if pred2 in self.inverted_predictors:
                    bin_k = (data_x[s, k] <= combined_thresholds[i])
                else:
                    bin_k = (data_x[s, k] >= combined_thresholds[i])
                if bin_j and bin_k:
                    bin_x[s, i] = 1
                else:
                    bin_x[s, i] = 0
        return bin_x
