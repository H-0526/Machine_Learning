import os
import argparse
import random

import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, log_loss
from classic_models import find_model_using_name  # 导入find_model_using_name函数

# 设置随机种子函数
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("Random seed has been set to:", seed)

# 命令行参数解析
parser = argparse.ArgumentParser(description="Train wine quality classification models.")
parser.add_argument('--model', type=str, default='logistic_regression', choices=['logistic_regression', 'naive_bayes'],
                    help="Model to use for classification.")
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
def create_logger(model_name, cv_folds):
    # 创建 results_raw 文件夹
    if not os.path.exists('results_preprocessed'):
        os.makedirs('results_preprocessed')

    # 设置日志文件名
    log_file = f"results_preprocessed/{model_name}_cv{cv_folds}_wine_quality.log"

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

# 主函数
def main():
    X, y = load_data()

    # 获取模型
    model = find_model_using_name(args.model)

    # 创建日志记录器
    logger = create_logger(args.model, args.cv_folds)

    # 记录模型开始训练
    logger.info(f"Starting training for model: {args.model}")

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 使用class_weight='balanced'来处理类别不平衡
    if hasattr(model, 'model'):  # 处理 LogisticRegressionModel 等模型
        model.model.class_weight = 'balanced'
    elif hasattr(model, 'class_weight'):  # 如果模型本身有 class_weight 参数
        model.class_weight = 'balanced'

    # 使用分层交叉验证
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f}")

    # 在交叉验证后训练和预测
    model.fit(X_train, y_train)  # 显式训练模型，确保它已被训练，使用 fit 方法
    y_pred_proba = model.predict_proba(X_test)  # 获取预测的概率分布

    # 计算负对数损失
    loss = log_loss(y_test, y_pred_proba)
    logger.info(f"Negative Log-Loss: {loss:.4f}")

    # 使用 zero_division=1 处理 Precision 为 NaN 的情况
    class_report = classification_report(y_test, model.predict(X_test), zero_division=1)
    logger.info(f"Classification Report:\n{class_report}")

    # 记录测试集准确率
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    # 计算容差分类准确率
    tolerance_acc_10 = tolerance_accuracy(y_test, model.predict(X_test), tolerance=1.0)

    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Tolerance Accuracy (T=1.0): {tolerance_acc_10:.4f}")

    # 记录类别分布
    logger.info(f"Class distribution:\n{pd.Series(y).value_counts()}")

    # 打印到控制台
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Negative Log-Loss: {loss:.4f}")
    print(f"Tolerance Accuracy (T=1.0): {tolerance_acc_10:.4f}")
    print(f"Classification Report:\n{class_report}")

if __name__ == "__main__":
    main()
