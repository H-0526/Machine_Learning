import os
import argparse
import random
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from regression_models import find_model_using_name  # 假设这里已经导入了 CubistModel

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed has been set to: {seed}")


# 数据加载
def load_data(file_path):
    """
    加载 CSV 数据
    :param file_path: str, 数据文件路径
    :return: tuple, 特征矩阵和目标变量
    """
    try:
        data = pd.read_csv(file_path)  # 加载 CSV 数据
    except Exception as e:
        raise ValueError(f"Failed to load CSV file. Ensure the file is valid: {e}")

    X = data.iloc[:, :-1].values  # 转换为 NumPy 数组
    y = data.iloc[:, -1].values  # 转换为 NumPy 数组
    return X, y


# 创建日志记录器
def create_logger(model_name, cv_folds):
    log_dir = "results_preprocessed"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{model_name}_regression_cv{cv_folds}_compressive_strength.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


# 记录模型超参数
def log_model_hyperparameters(logger, model):
    if hasattr(model, "get_hyperparameters"):
        hyperparameters = model.get_hyperparameters()
        logger.info("Model Hyperparameters:")
        for key, value in hyperparameters.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.warning("Model does not provide hyperparameters.")


# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train regression models using Cubist.")
    parser.add_argument('--model', type=str, default='cubist', help="Model name to use.")
    parser.add_argument('--data_path', type=str, default=r'G:\Pycharm\Machine_Learning\data\processed_data\standardized_selected_features_data_concrete.csv',
                        help="Path to the dataset file.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--test_size', type=float, default=0.3, help="Proportion of data used for testing.")
    parser.add_argument('--cv_folds', type=int, default=5, help="Number of folds for cross-validation.")
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载数据
    try:
        X, y = load_data(args.data_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    # 初始化模型
    model = find_model_using_name(args.model)

    # 创建日志记录器
    logger = create_logger(args.model, args.cv_folds)
    logger.info(f"Training {args.model} model with {args.cv_folds}-fold cross-validation.")

    # 记录模型超参数
    log_model_hyperparameters(logger, model)

    # 使用交叉验证
    try:
        kfold = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(model.model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        cv_scores = -cv_scores  # 转换为正值
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        print(f"Cross-validation failed: {e}")
        return

    # 记录交叉验证结果
    mean_cv_score = np.mean(cv_scores)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score (MSE): {mean_cv_score:.4f}")

    # 训练和评估模型
    model.train(X_train, y_train)
    mse, rmse, mae, r2 = model.evaluate(X_test, y_test)

    # 记录测试集性能
    logger.info(f"Mean Squared Error: {mse:.4f}")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"R-squared (R^2): {r2:.4f}")

    # 打印测试结果
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared (R^2): {r2:.4f}")


if __name__ == "__main__":
    main()
