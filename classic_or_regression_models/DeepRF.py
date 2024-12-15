import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np


class DeepNN(nn.Module):
    """
    A simple deep neural network for feature extraction.
    """
    def __init__(self, input_size, output_size=1):
        super(DeepNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        return self.model(x)


class DeepRFModel(BaseEstimator):
    def __init__(self, input_size, task="classification"):
        self.input_size = input_size
        self.task = task.lower()
        output_size = 1  # 二分类或回归输出一个值

        # Initialize neural network
        self.nn_model = DeepNN(input_size, output_size)

        # Initialize random forest
        if self.task == "classification":
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        elif self.task == "regression":
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

    def fit(self, X_train, y_train, nn_epochs=10, nn_batch_size=32, learning_rate=1e-3):
        """
        Train the DeepRF model.
        """
        # 确保 X_train 和 y_train 的格式兼容
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        # 在回归任务中，确保 y_train 的形状为 (N, 1)
        if self.task == "regression":
            y_train = y_train.unsqueeze(1)

        # Optimizer and loss function
        optimizer = optim.Adam(self.nn_model.parameters(), lr=learning_rate)
        loss_function = nn.BCEWithLogitsLoss() if self.task == "classification" else nn.MSELoss()

        # Train neural network
        self.nn_model.train()
        for epoch in range(nn_epochs):
            permutation = torch.randperm(X_train.size(0))
            for i in range(0, X_train.size(0), nn_batch_size):
                indices = permutation[i:i + nn_batch_size]
                batch_x = X_train[indices]
                batch_y = y_train[indices]

                # 确保分类任务的目标形状与输出一致
                if self.task == "classification" and batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)

                optimizer.zero_grad()
                outputs = self.nn_model(batch_x)
                loss = loss_function(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Extract features
        self.nn_model.eval()
        with torch.no_grad():
            deep_features = self.nn_model(X_train).detach().numpy()

        # Train random forest
        self.rf_model.fit(deep_features, y_train.numpy().ravel())

    def predict(self, X):
        # 确保输入格式正确
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        # Extract features
        self.nn_model.eval()
        with torch.no_grad():
            deep_features = self.nn_model(X).detach().numpy()

        # Predict using random forest
        return self.rf_model.predict(deep_features)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        # Extract features
        self.nn_model.eval()
        with torch.no_grad():
            deep_features = self.nn_model(X).detach().numpy()

        if self.task == "classification":
            return self.rf_model.predict_proba(deep_features)
        else:
            raise ValueError("predict_proba is not supported for regression tasks")

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        if self.task == "classification":
            return accuracy_score(y_test, y_pred)
        elif self.task == "regression":
            return mean_squared_error(y_test, y_pred)

    def get_hyperparameters(self):
        """Return the model's hyperparameters."""
        return {
            "task": self.task,
            "random_forest_params": self.rf_model.get_params(),
            "nn_params": {
                "hidden_layers": [64, 32, 16],
                "dropout": 0.1,
                "activation": "ReLU"
            }
        }

    def set_params(self, **params):
        """Set the model parameters for compatibility with Scikit-learn."""
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_params(self, deep=True):
        """Get the model parameters for compatibility with Scikit-learn."""
        return {"input_size": self.input_size, "task": self.task}
