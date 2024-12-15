from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.base import BaseEstimator


class GBDTModel(BaseEstimator):
    """
    A unified Gradient Boosting model class for classification and regression tasks.
    """
    def __init__(self, task="regression", **gbdt_params):
        """
        Initialize the Gradient Boosting model with configurable hyperparameters.

        :param task: Type of task ("classification" or "regression").
        :param gbdt_params: Additional parameters for the Gradient Boosting model.
        """
        self.task = task.lower()
        self.gbdt_params = gbdt_params

        # Configure Gradient Boosting model
        if self.task == "classification":
            self.model = GradientBoostingClassifier(
                n_estimators=100,       # Default number of boosting stages
                learning_rate=0.1,      # Default learning rate
                max_depth=3,            # Default maximum depth of individual trees
                subsample=1.0,          # Default subsample fraction
                random_state=42,        # Default random state
                **self.gbdt_params
            )
        elif self.task == "regression":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                subsample=1.0,
                random_state=42,
                **self.gbdt_params
            )
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

    def fit(self, X_train, y_train):
        """
        Train the Gradient Boosting model.

        :param X_train: Training feature matrix.
        :param y_train: Training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Make predictions using the Gradient Boosting model.

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
            "gbdt_params": self.model.get_params()
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
                self.gbdt_params[param] = value

        # Reinitialize the model with updated parameters
        self.__init__(task=self.task, **self.gbdt_params)
        return self

    def get_params(self, deep=True):
        """
        Get the model parameters for compatibility with Scikit-learn.

        :param deep: Whether to return deep parameters.
        :return: Dictionary of model parameters.
        """
        return {"task": self.task, **self.gbdt_params}


# Example usage: Unified training file
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split

    # Example: Classification task
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    clf_model = GBDTModel(task="classification", n_estimators=200, max_depth=5)
    clf_model.fit(X_train_clf, y_train_clf)
    accuracy = clf_model.evaluate(X_test_clf, y_test_clf)
    print("Classification Accuracy:", accuracy)

    # Example: Regression task
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_model = GBDTModel(task="regression", n_estimators=200, learning_rate=0.05)
    reg_model.fit(X_train_reg, y_train_reg)
    mse = reg_model.evaluate(X_test_reg, y_test_reg)
    print("Regression MSE:", mse)
