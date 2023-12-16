from sklearn.linear_model import LogisticRegression
from read_data import read_data
from mixed_model import MixedModel

data_x, data_y = read_data(["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"], "Dead")

logist_reg = LogisticRegression()
logist_reg.fit(data_x, data_y)
y_pred = logist_reg.predict_proba(data_x)[:, 1]

selected_features = [0, 1, 2]
omega = 0.6
mixed_model = MixedModel(selected_features, omega, logist_reg.coef_, logist_reg.intercept_)
