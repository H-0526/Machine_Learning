import numpy as np
from rulefit import RuleFit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score


class CubistModel:
    """
    A unified Cubist model class with internal hyperparameter configuration
    for regression tasks.
    """
    def __init__(self):
        """
        Initialize the Cubist model with default hyperparameters.
        """
        self.model = RuleFit(
            tree_generator=None,  # Use default tree generator
            max_rules=200,        # Maximum number of rules
            tree_size=4,          # Maximum depth of trees
            sample_fract=0.7,     # Fraction of samples for fitting rules
            memory_par=0.01,      # Regularization for linear terms
            random_state=42,      # For reproducibility
            max_iter = 1000   # Maximum iterations for LassoCV
        )

    def train(self, X_train, y_train):
        """
        Train the Cubist model.

        :param X_train: Training feature matrix.
        :param y_train: Training labels.
        """
        self.model.fit(X_train, y_train, feature_names=None)

    def predict(self, X):
        """
        Make predictions using the Cubist model.

        :param X: Feature matrix for prediction.
        :return: Predicted values.
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance.

        :param X_test: Test feature matrix.
        :param y_test: True labels for the test set.
        :return: A tuple containing Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
                 Mean Absolute Error (MAE), and R-squared (R^2).
        """
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, rmse, mae, r2  # 返回元组


# Example usage: Unified training file
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    # Generate synthetic regression dataset
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Cubist model
    cubist_model = CubistModel()
    cubist_model.train(X_train, y_train)
    mse = cubist_model.evaluate(X_test, y_test)
    print("Cubist Regression MSE:", mse)