from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.base import BaseEstimator


class LightGBM(BaseEstimator):
    """
    一个统一的 LightGBM 模型类，可用于分类和回归任务。
    """
    def __init__(self, task="classification", **lgbm_params):
        """
        初始化 LightGBM 模型，并支持用户配置超参数。

        :param task: 任务类型 ("classification" 或 "regression")。
        :param lgbm_params: LightGBM 模型的额外参数。
        """
        self.task = task.lower()
        self.lgbm_params = lgbm_params

        # 根据任务类型选择模型
        if self.task == "classification":
            # 分类任务启用类别权重自动平衡
            self.model = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                boosting_type="gbdt",
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                class_weight="balanced",  # 启用类别权重平衡
                **self.lgbm_params
            )
        elif self.task == "regression":
            # 回归任务配置
            self.model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=-1,
                num_leaves=31,
                boosting_type="gbdt",
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                **self.lgbm_params
            )
        else:
            raise ValueError("任务类型必须为 'classification' 或 'regression'")

    def fit(self, X_train, y_train, eval_set=None, early_stopping_rounds=None):
        """
        训练 LightGBM 模型。

        :param X_train: 训练特征矩阵。
        :param y_train: 训练标签。
        :param eval_set: 用于早停的验证集 (可选)。
        :param early_stopping_rounds: 早停轮次 (可选)。
        """
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
        使用 LightGBM 模型进行预测。

        :param X: 用于预测的特征矩阵。
        :return: 预测结果。
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        返回分类任务的预测概率。

        :param X: 用于预测的特征矩阵。
        :return: 分类任务的预测概率。
        """
        if self.task == "classification":
            return self.model.predict_proba(X)
        else:
            raise AttributeError("predict_proba 仅适用于分类任务")

    def evaluate(self, X_test, y_test):
        """
        评估模型性能。

        :param X_test: 测试特征矩阵。
        :param y_test: 测试集的真实标签。
        :return: 分类任务的准确率或回归任务的均方误差 (MSE)。
        """
        y_pred = self.predict(X_test)
        if self.task == "classification":
            return accuracy_score(y_test, y_pred)
        elif self.task == "regression":
            return mean_squared_error(y_test, y_pred)

    def get_hyperparameters(self):
        """
        获取模型的超参数，用于记录或调试。

        :return: 模型超参数字典。
        """
        return {
            "task": self.task,
            "lgbm_params": self.model.get_params()
        }

    def set_params(self, **params):
        """
        设置模型参数，以与 Scikit-learn 的接口兼容。

        :param params: 要设置的模型参数。
        """
        for param, value in params.items():
            if param == "task":
                self.task = value
            else:
                self.lgbm_params[param] = value

        # 使用更新后的参数重新初始化模型
        self.__init__(task=self.task, **self.lgbm_params)
        return self

    def get_params(self, deep=True):
        """
        获取模型参数，以与 Scikit-learn 的接口兼容。

        :param deep: 是否返回深层次参数。
        :return: 模型参数字典。
        """
        return {"task": self.task, **self.lgbm_params}


# 示例用法
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split

    # 示例：分类任务
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    clf_model = LightGBM(task="classification", n_estimators=200, max_depth=5)
    clf_model.fit(X_train_clf, y_train_clf)
    accuracy = clf_model.evaluate(X_test_clf, y_test_clf)
    print("分类任务准确率:", accuracy)

    # 示例：回归任务
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_model = LightGBM(task="regression", n_estimators=200, learning_rate=0.05)
    reg_model.fit(X_train_reg, y_train_reg)
    mse = reg_model.evaluate(X_test_reg, y_test_reg)
    print("回归任务均方误差 (MSE):", mse)
