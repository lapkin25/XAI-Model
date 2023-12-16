from sklearn.linear_model import LogisticRegression
from read_data import read_data
from mixed_model import MixedModel

predictors = ["Age", "HR", "Killip class", "Cr", "EF LV", "NEUT", "EOS", "PCT", "Glu", "SBP"]
data_x, data_y = read_data(predictors, "Dead")

logist_reg = LogisticRegression()
logist_reg.fit(data_x, data_y)
y_pred = logist_reg.predict_proba(data_x)[:, 1]

print("Параметры логистической регрессии:")
print("Веса:", logist_reg.coef_.ravel())
print("Смещение:", logist_reg.intercept_[0])
print()

selected_features = [0, 1, 2]
omega = 0 # 0.6
mixed_model = MixedModel(selected_features, omega, logist_reg.coef_.ravel(), logist_reg.intercept_[0])
mixed_model.fit(data_x, data_y)

print("Параметры смешанной модели:")
print("omega =", omega * 100, "%")
print("Выбранные предикторы:", map(lambda i: predictors[i], selected_features))
print("Веса многофакторной модели:", mixed_model.weights)
print("Смещение многофакторной модели:", mixed_model.intercept)
print("Весовые коэффициенты при однофакторных моделях:", mixed_model.omega_coefficients * 100)
