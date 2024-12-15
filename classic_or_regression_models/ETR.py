from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.base import BaseEstimator


class ETRModel(BaseEstimator):
    """
    A unified ExtraTrees model class for classification and regression tasks.
    """
    def __init__(self, task="regression", **extra_trees_params):
        """
        Initialize the ExtraTrees model with configurable hyperparameters.

        :param task: Type of task ("classification" or "regression").
        :param extra_trees_params: Additional parameters for the ExtraTrees model.
        """
        self.task = task.lower()
        self.extra_trees_params = extra_trees_params

        # Configure ExtraTrees model
        if self.task == "classification":
            self.model = ExtraTreesClassifier(
                n_estimators=100,  # Default number of trees
                max_depth=None,    # No limit on tree depth
                min_samples_split=2,  # Default minimum samples to split
                random_state=42,   # For reproducibility
                **self.extra_trees_params
            )
        elif self.task == "regression":
            self.model = ExtraTreesRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                **self.extra_trees_params
            )
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

    def fit(self, X_train, y_train):
        """
        Train the ExtraTrees model.

        :param X_train: Training feature matrix.
        :param y_train: Training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Make predictions using the ExtraTrees model.

        :param X: Feature matrix for prediction.
        :return: Predicted values.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Return predicted probabilities for classification tasks.

        :param X: Feature matrix for prediction.
        :return: Predicted probabilities.
        """
        if self.task == "classification":
            return self.model.predict_proba(X)
        else:
            raise AttributeError("predict_proba is only available for classification tasks")

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance.

        :param X_test: Test feature matrix.
        :param y_test: True labels for the test set.
        :return: Accuracy for classification or Mean Squared Error (MSE) for regression.
        """
        y_pred = self.predict(X_test)
        if self.task == "classification":
            return accuracy_score(y_test, y_pred)
        elif self.task == "regression":
            return mean_squared_error(y_test, y_pred)

    def get_hyperparameters(self):
        """
        Return the model's hyperparameters for logging or debugging purposes.
        """
        return {
            "task": self.task,
            "extra_trees_params": self.model.get_params()
        }

    def set_params(self, **params):
        """
        Set the model parameters for compatibility with Scikit-learn.

        :param params: Model parameters to set.
        """
        for param, value in params.items():
            if param == "task":
                self.task = value
            else:
                self.extra_trees_params[param] = value

        # Reinitialize the model with updated parameters
        self.__init__(task=self.task, **self.extra_trees_params)
        return self

    def get_params(self, deep=True):
        """
        Get the model parameters for compatibility with Scikit-learn.

        :param deep: Whether to return deep parameters.
        :return: Dictionary of model parameters.
        """
        return {"task": self.task, **self.extra_trees_params}
