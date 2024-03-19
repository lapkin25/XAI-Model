class NewCombinedFeaturesModel:
    def __init__(self, verbose_training=False, p0=0.05, K=10):
        self.cutoffs = None
        self.weights = None
        self.intercept = None
        self.combined_features = None  # список троек (k, j, xj_cutoff)
        self.combined_weights = None

        self.verbose_training = verbose_training
        self.p0 = p0
        self.K = K


    def fit(self, x, y):


    # возвращает для каждой точки две вероятности: "0" и "1"
    def predict_proba(self, x, y):
