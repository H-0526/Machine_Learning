from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

class LogisticRegressionModel(BaseEstimator, ClassifierMixin):
    """
    A unified Logistic Regression model class with default hyperparameter settings.
    """
    def __init__(self):
        """
        Initialize the Logistic Regression model with default parameters.
        """
        self.model = LogisticRegression(
            penalty="l2",       # Regularization
            C=1,                # Inverse of regularization strength
            solver="lbfgs",     # Solver for multi-class classification
            max_iter=200,       # Maximum number of iterations
            class_weight="balanced",  # Automatically adjust weights for unbalanced classes
            random_state=42     # Random state for reproducibility
        )

    def fit(self, X, y):
        """
        Train the Logistic Regression model.

        :param X: Training feature matrix.
        :param y: Training labels.
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_  # Ensuring that classes_ is set
        return self

    def predict(self, X):
        """
        Make predictions using the Logistic Regression model.

        :param X: Feature matrix for prediction.
        :return: Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities using the Logistic Regression model.

        :param X: Feature matrix for prediction.
        :return: Predicted probabilities for each class.
        """
        return self.model.predict_proba(X)  # Return predicted probabilities

    def score(self, X, y):
        """
        Return the accuracy of the model.

        :param X: Test feature matrix.
        :param y: True labels for the test set.
        :return: Accuracy score.
        """
        return self.model.score(X, y)
