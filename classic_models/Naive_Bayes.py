from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.base import BaseEstimator, ClassifierMixin

class NaiveBayesModel(BaseEstimator, ClassifierMixin):
    """
    A unified Naive Bayes model class with internal hyperparameter configuration
    for Bernoulli, Multinomial, and Gaussian classifiers.
    """
    def __init__(self):
        """
        Initialize the Naive Bayes model without setting the model type.
        """
        self.model_type = None
        self.model = None
        self.hyperparameters = {}  # Dictionary to store hyperparameters

    def set_model_type(self, model_type="bernoulli"):
        """
        Set the model type after initialization.

        :param model_type: Type of Naive Bayes classifier ("bernoulli", "multinomial", "gaussian").
        """
        self.model_type = model_type.lower()
        self.hyperparameters = {}  # Reset hyperparameters

        if self.model_type == "bernoulli":
            # Hyperparameters for BernoulliNB
            self.hyperparameters = {
                'alpha': 1.0,        # Smoothing parameter
                'binarize': 0.5,     # Threshold for binarizing features
                'fit_prior': True    # Learn class priors
            }
            self.model = BernoulliNB(**self.hyperparameters)
        elif self.model_type == "multinomial":
            # Hyperparameters for MultinomialNB
            self.hyperparameters = {
                'alpha': 1.0,        # Smoothing parameter
                'fit_prior': True    # Learn class priors
            }
            self.model = MultinomialNB(**self.hyperparameters)
        elif self.model_type == "gaussian":
            # Hyperparameters for GaussianNB
            self.hyperparameters = {
                'priors': None,          # Class priors (None means learn from raw_data)
                'var_smoothing': 1e-7    # Stability parameter for variance
            }
            self.model = GaussianNB(**self.hyperparameters)
        else:
            raise ValueError("model_type must be 'bernoulli', 'multinomial', or 'gaussian'")

    def fit(self, X, y):
        """
        Train the Naive Bayes model.
        """
        if self.model is None:
            raise ValueError("Model has not been set. Use set_model_type() to initialize the model.")
        self.model.fit(X, y)

    def predict(self, X):
        """
        Make predictions using the Naive Bayes model.
        """
        if self.model is None:
            raise ValueError("Model has not been set. Use set_model_type() to initialize the model.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities using the Naive Bayes model.
        """
        if self.model is None:
            raise ValueError("Model has not been set. Use set_model_type() to initialize the model.")
        return self.model.predict_proba(X)

    def score(self, X, y):
        """
        Return the accuracy of the model.
        """
        if self.model is None:
            raise ValueError("Model has not been set. Use set_model_type() to initialize the model.")
        return self.model.score(X, y)

    def get_params(self, deep=True):
        """
        获取模型参数，兼容 Scikit-learn API。
        """
        return {"model_type": self.model_type}

    def set_params(self, **params):
        """
        设置模型参数，兼容 Scikit-learn API。
        """
        if "model_type" in params:
            self.set_model_type(params["model_type"])
        return self

