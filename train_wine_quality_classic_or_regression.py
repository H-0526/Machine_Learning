import os
import argparse
import random
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, log_loss, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from classic_or_regression_models import find_model_using_name  # 导入find_model_using_name函数
from classic_or_regression_models.DeepRF import DeepRFModel  # 导入 DeepRFModel
from imblearn.over_sampling import SMOTE  # 导入 SMOTE


# 设置随机种子函数
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("Random seed has been set to:", seed)


# 命令行参数解析
parser = argparse.ArgumentParser(description="Train wine quality classification or regression models.")
parser.add_argument('--model', type=str, default='xgboost',
                    choices=['deep_rf', 'etr', 'gbdt', 'lightgbm', 'rf', 'svm', 'xgboost'],
                    help="Model to use for training.")
parser.add_argument('--task', type=str, default='classification', choices=['classification', 'regression'],
                    help="Task to perform ('classification' or 'regression').")
parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument('--test_size', type=float, default=0.3, help="Proportion of raw_data used for testing.")
parser.add_argument('--cv_folds', type=int, default=5, help="Number of folds for cross-validation.")
args = parser.parse_args()

# 设置随机种子
set_seed(args.seed)


# 数据加载与预处理
def load_data(data_path= r'G:\Pycharm\Machine_Learning\data\processed_data\standardized_selected_features_data_combined_wine.csv'):
    # 加载数据
    data = pd.read_csv(data_path)  # 从 CSV 文件加载数据
    X = data.iloc[:, :-1]  # 自变量
    y = data.iloc[:, -1]  # 目标变量

    # 将目标变量转换为整数标签，避免分类任务中目标变量被误认为连续值
    target_scaler_path = r"G:\Pycharm\Machine_Learning\data\processed_data\scalers\target_scaler_selected_features_data_combined_wine.joblib"
    target_scaler = joblib.load(target_scaler_path)
    y = target_scaler.inverse_transform(y.values.reshape(-1, 1)).flatten().astype(int)  # 还原分类标签

    return X, y


# 创建日志文件
def create_logger(model_name, task, cv_folds):
    if not os.path.exists('results_preprocessed'):
        os.makedirs('results_preprocessed')

    # 设置日志文件名，包含 model_name 和 task
    log_file = f"results_preprocessed/{model_name}_{task}_cv{cv_folds}_wine_quality.log"

    # 配置日志格式
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 添加文件处理器
    logger.addHandler(file_handler)

    return logger

def tolerance_accuracy(y_true, y_pred, tolerance=1):
    """
    计算容差分类准确率。
    T=1.0 相当于为多分类任务设置了一个“±1”范围的误差容忍度
    :param y_true: 实际类别。
    :param y_pred: 预测类别。
    :param tolerance: 容差范围。
    :return: 容差准确率。
    """
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if abs(true - pred) <= tolerance:
            correct += 1
    return correct / len(y_true)

# 手动交叉验证函数
def manual_cross_val_score(model, X, y, cv_folds=5, scoring='accuracy', task='classification'):
    """
    手动交叉验证函数，支持 DeepRFModel。
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.seed)
    scores = []

    # 将 y 转为 NumPy 数组
    X = np.array(X)
    y = np.array(y)

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if args.model == 'deep_rf':
            input_size = X_train.shape[1]
            model = DeepRFModel(input_size=input_size, task=task)

        model.fit(X_train, y_train)

        if task == 'classification':
            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
        elif task == 'regression':
            y_pred = model.predict(X_val)
            score = mean_squared_error(y_val, y_pred)

        scores.append(score)

    return np.array(scores)


def log_model_hyperparameters(logger, model):
    """
    Log the hyperparameters of the model.

    :param logger: Logger instance.
    :param model: Model instance with `get_hyperparameters` method.
    """
    if hasattr(model, "get_hyperparameters"):
        hyperparameters = model.get_hyperparameters()
        logger.info("Model Hyperparameters:")
        for key, value in hyperparameters.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.warning("Model does not provide hyperparameters.")

# 主函数
def main():
    X, y = load_data()

    # 获取模型
    if args.model == 'deep_rf':
        input_size = X.shape[1]
        model = DeepRFModel(input_size=input_size, task=args.task)
    else:
        model = find_model_using_name(args.model, task=args.task)

    # 创建日志记录器
    logger = create_logger(args.model, args.task, args.cv_folds)

    # 记录模型开始训练和超参数信息
    logger.info(f"Starting training for {args.model} model with {args.task} task.")
    logger.info(f"Using {args.cv_folds}-fold cross-validation.")

    # 记录模型超参数
    log_model_hyperparameters(logger, model)

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    # # 应用 SMOTE 进行过采样，但实验效果不好, 不建议采用
    # if args.task == "classification":
    #     smote = SMOTE(random_state=args.seed)
    #     # 自动计算适用于多分类任务的采样策略
    #     # 以采样至主要类别数量的 25% 为例
    #     class_counts = np.bincount(y_train)
    #     max_class_count = max(class_counts)
    #     target_class_counts = { 4: 1500, 7: 1500 }
    #
    #     smote = SMOTE(sampling_strategy=target_class_counts, random_state=args.seed)
    #     X_train, y_train = smote.fit_resample(X_train, y_train)
    #     logger.info(
    #         f"Applied SMOTE with sampling_strategy={target_class_counts}. Resampled class distribution: {np.bincount(y_train)}")

    # 手动交叉验证
    if args.model == 'deep_rf' or 'etr':
        cv_scores = manual_cross_val_score(model, X_train, y_train, cv_folds=args.cv_folds, task=args.task)
    else:
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy' if args.task == 'classification' else 'neg_mean_squared_error')

    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score: {cv_scores.mean():.4f}")

    # 训练和评估模型
    model.fit(X_train, y_train)

    if args.task == 'classification':
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        loss = log_loss(y_test, y_pred_proba)
        class_report = classification_report(y_test, model.predict(X_test), zero_division=1)
        test_accuracy = accuracy_score(y_test, y_pred)

        # 计算容差分类准确率
        tolerance_acc_10 = tolerance_accuracy(y_test, y_pred, tolerance=1.0)

        logger.info(f"Negative Log-Loss: {loss:.4f}")
        logger.info(f"Classification Report:\n{class_report}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Tolerance Accuracy (T=1.0): {tolerance_acc_10:.4f}")

        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Negative Log-Loss: {loss:.4f}")
        print(f"Tolerance Accuracy (T=1.0): {tolerance_acc_10:.4f}")
        print(f"Classification Report:\n{class_report}")
    else:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Mean Squared Error: {mse:.4f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
        logger.info(f"R-squared (R²): {r2:.4f}")

        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R²): {r2:.4f}")


if __name__ == "__main__":
    main()
