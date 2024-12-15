from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator


class XGBoostModel(BaseEstimator):
    """
    A unified XGBoost model class for classification and regression tasks.
    """
    def __init__(self, task="classification", **xgb_params):
        """
        Initialize the XGBoost model with configurable hyperparameters.

        :param task: Type of task ("classification" or "regression").
        :param xgb_params: Additional parameters for the XGBoost model.
        """
        self.task = task.lower()
        self.xgb_params = xgb_params
        self.label_encoder = None  # Initialize label encoder for classification tasks

        if self.task == "classification":
            # Default objective for classification
            if "objective" not in self.xgb_params:
                self.xgb_params["objective"] = "multi:softprob"

            self.model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                **self.xgb_params
            )
        elif self.task == "regression":
            self.model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                **self.xgb_params
            )
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

    def fit(self, X_train, y_train, eval_set=None, early_stopping_rounds=None):
        """
        Train the XGBoost model.

        :param X_train: Training feature matrix.
        :param y_train: Training labels.
        :param eval_set: Evaluation set for early stopping (optional).
        :param early_stopping_rounds: Number of rounds for early stopping (optional).
        """
        # Encode labels for classification
        if self.task == "classification":
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(y_train)

            # If eval_set is provided, encode labels in eval_set
            if eval_set:
                eval_set = [(X, self.label_encoder.transform(y)) for X, y in eval_set]

        if eval_set and early_stopping_rounds:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Make predictions using the XGBoost model.

        :param X: Feature matrix for prediction.
        :return: Predicted values.
        """
        y_pred = self.model.predict(X)

        # Decode predictions for classification
        if self.task == "classification" and self.label_encoder:
            y_pred = self.label_encoder.inverse_transform(y_pred)

        return y_pred

    def predict_proba(self, X):
        """
        Predict probability estimates for classification tasks.

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
            "xgb_params": self.model.get_params()
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
                self.xgb_params[param] = value

        # Reinitialize the model with updated parameters
        self.__init__(task=self.task, **self.xgb_params)
        return self

    def get_params(self, deep=True):
        """
        Get the model parameters for compatibility with Scikit-learn.

        :param deep: Whether to return deep parameters.
        :return: Dictionary of model parameters.
        """
        return {"task": self.task, **self.xgb_params}


# Example usage: Unified training file
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split

    # Example: Classification task
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, random_state=42, n_classes=5)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    clf_model = XGBoostModel(task="classification", n_estimators=200, max_depth=4)
    clf_model.fit(X_train_clf, y_train_clf)
    accuracy = clf_model.evaluate(X_test_clf, y_test_clf)
    print("Classification Accuracy:", accuracy)

    # Example: Regression task
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_model = XGBoostModel(task="regression", learning_rate=0.05)
    reg_model.fit(X_train_reg, y_train_reg)
    mse = reg_model.evaluate(X_test_reg, y_test_reg)
    print("Regression MSE:", mse)
