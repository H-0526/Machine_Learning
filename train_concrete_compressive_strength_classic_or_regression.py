import os
import argparse
import random
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from classic_or_regression_models import find_model_using_name
from classic_or_regression_models.DeepRF import DeepRFModel


# ============ 参数解析 ============
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train concrete compressive strength regression models.")
    parser.add_argument('--model', type=str, default='lightgbm',
                        choices=['deep_rf', 'etr', 'gbdt', 'lightgbm', 'rf', 'svm', 'xgboost'],
                        help="Model to use for training.")
    parser.add_argument('--task', type=str, default='regression',
                        choices=['classification', 'regression'],
                        help="Task to perform ('classification' or 'regression').")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--test_size', type=float, default=0.3, help="Proportion of raw_data used for testing.")
    parser.add_argument('--cv_folds', type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument('--nn_epochs', type=int, default=10, help="Number of epochs for training the neural network.")
    parser.add_argument('--nn_batch_size', type=int, default=32, help="Batch size for neural network training.")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate for neural network training.")
    return parser.parse_args()


# ============ 随机种子设置 ============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("Random seed has been set to:", seed)


# ============ 日志记录器 ============
def create_logger(model_name, task, cv_folds):
    if not os.path.exists('results_preprocessed'):
        os.makedirs('results_preprocessed')
    log_file = f"results_preprocessed/{model_name}_{task}_cv{cv_folds}_compressive_strength.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


# ============ 数据加载与归一化 ============
def load_and_preprocess_data(data_path= r'G:\Pycharm\Machine_Learning\data\processed_data\standardized_selected_features_data_concrete.csv'):
    """
    加载数据集并返回特征和目标变量
    :param data_path: str, 数据集路径
    :return: X (特征), y (目标变量)
    """
    # 数据加载
    data = pd.read_csv(data_path)  # 从 CSV 文件加载数据
    X = data.iloc[:, :-1]  # 自变量
    y = data.iloc[:, -1]  # 目标变量

    return X, y

# ============ 确保目标变量形状一致 ============
def ensure_y_shape(y):
    return np.array(y).flatten()


# ============ 手动交叉验证 ============
def manual_cross_val_score(model, X, y, cv_folds, task, logger):
    # 确保 X 和 y 是 numpy 数组
    X, y = np.array(X), np.array(y)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.seed) if task == "classification" \
        else KFold(n_splits=cv_folds, shuffle=True, random_state=args.seed)

    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 模型训练
        if isinstance(model, DeepRFModel):
            model.fit(X_train, y_train,
                      nn_epochs=args.nn_epochs,
                      nn_batch_size=args.nn_batch_size,
                      learning_rate=args.learning_rate)
        else:
            model.fit(X_train, y_train)

        # 模型预测与得分计算
        y_pred = model.predict(X_val)
        if task == 'regression':
            mse = mean_squared_error(y_val, y_pred)
            scores.append(mse)
            logger.info(f"Fold {fold_idx + 1}: MSE = {mse:.4f}")
        elif task == 'classification':
            acc = accuracy_score(y_val, y_pred)
            scores.append(acc)
            logger.info(f"Fold {fold_idx + 1}: Accuracy = {acc:.4f}")

    return scores


# ============ 记录模型超参数 ============
def log_model_hyperparameters(logger, model):
    if hasattr(model, "get_hyperparameters"):
        hyperparameters = model.get_hyperparameters()
        logger.info("Model Hyperparameters:")
        for key, value in hyperparameters.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.warning("Model does not provide hyperparameters.")


# ============ 主训练函数 ============
def main():
    # 解析参数和设置随机种子
    global args
    args = parse_arguments()
    set_seed(args.seed)

    # 加载与预处理数据
    X, y = load_and_preprocess_data()
    y = ensure_y_shape(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    # 初始化模型
    if args.model == 'deep_rf':
        model = DeepRFModel(input_size=X.shape[1], task=args.task)
    else:
        model = find_model_using_name(args.model, task=args.task)

    # 创建日志记录器
    logger = create_logger(args.model, args.task, args.cv_folds)
    logger.info(f"Training {args.model} model for {args.task} task with {args.cv_folds}-fold cross-validation.")

    # 记录模型超参数
    log_model_hyperparameters(logger, model)

    # 手动交叉验证逻辑
    if args.model in ['deep_rf', 'etr', 'gbdt', 'lightgbm', 'rf', 'svm', 'xgboost']:
        logger.info("Using manual cross-validation...")
        cv_scores = manual_cross_val_score(model, X_train, y_train, cv_folds=args.cv_folds, task=args.task, logger=logger)
    else:
        logger.info("Using sklearn's cross_val_score for cross-validation...")
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed) \
            if args.task == "classification" else KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv,
            scoring='accuracy' if args.task == 'classification' else 'neg_mean_squared_error'
        )

    # 记录交叉验证结果
    mean_cv_score = np.mean(cv_scores)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score: {mean_cv_score:.4f}")

    # 模型训练
    logger.info("Fitting the model on the entire training set...")
    if isinstance(model, DeepRFModel):
        model.fit(X_train, y_train,
                  nn_epochs=args.nn_epochs,
                  nn_batch_size=args.nn_batch_size,
                  learning_rate=args.learning_rate)
    else:
        model.fit(X_train, y_train)

    # 模型评估
    y_pred = model.predict(X_test)
    if args.task == 'regression':
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Mean Squared Error: {mse:.4f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
        logger.info(f"R-squared (R^2): {r2:.4f}")

        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R^2): {r2:.4f}")

    elif args.task == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Test set Accuracy: {accuracy:.4f}")
        print(f"Test set Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
