from sklearn.preprocessing import PolynomialFeatures


class FeatureTransformer:

    def __init__(self, polynomial_degree=1):

        self.degree = polynomial_degree
        self.transformer = None

    def fit_transform(self, X):

        if self.degree <= 1:
            return X

        self.transformer = PolynomialFeatures(self.degree, include_bias=False)

        return self.transformer.fit_transform(X)

    def transform(self, X):

        if self.transformer:
            return self.transformer.transform(X)

        return X