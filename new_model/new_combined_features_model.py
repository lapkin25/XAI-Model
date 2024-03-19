class NewCombinedFeaturesModel:
    # p0 - порог отсечения
    # K - число комбинированных признаков
    # delta_a - параметр регуляризации для порога
    # delta_w - параметр регуляризации для веса
    def __init__(self, verbose_training=False, p0=0.05, K=10, delta_a=None, delta_w=None):
        self.cutoffs = None
        self.weights = None
        self.intercept = None
        self.combined_features = None  # список троек (k, j, xj_cutoff)
        self.combined_weights = None

        self.verbose_training = verbose_training
        self.p0 = p0
        self.K = K
        self.delta_a = delta_a
        self.delta_w = delta_w


    def fit(self, x, y):



    # возвращает для каждой точки две вероятности: "0" и "1"
    def predict_proba(self, x, y):
