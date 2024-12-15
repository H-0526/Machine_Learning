import os
import argparse
import random

import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, log_loss
from classic_models.Naive_Bayes import NaiveBayesModel  # 导入 NaiveBayesModel 类
from utils.lr_methods import warmup  # 导入warmup学习率调整方法

# 设置随机种子函数
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("Random seed has been set to:", seed)

# 命令行参数解析
parser = argparse.ArgumentParser(description="Train wine quality classification models.")
parser.add_argument('--model', type=str, default='naive_bayes', choices=['naive_bayes'],
                    help="Model to use for classification. (Currently only 'naive_bayes' is available.)")
parser.add_argument('--model_type', type=str, default='gaussian', choices=['bernoulli', 'multinomial', 'gaussian'],
                    help="Type of Naive Bayes classifier to use ('bernoulli', 'multinomial', 'gaussian').")
parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument('--test_size', type=float, default=0.3, help="Proportion of raw_data used for testing.")
parser.add_argument('--cv_folds', type=int, default=5, help="Number of folds for cross-validation.")
args = parser.parse_args()

# 设置随机种子
set_seed(args.seed)

# 数据加载
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
def create_logger(model_name, model_type, cv_folds):
    # 创建 results_raw 文件夹
    if not os.path.exists('results_preprocessed'):
        os.makedirs('results_preprocessed')

    # 设置日志文件名，包含 model_name 和 task
    log_file = f"results_preprocessed/{model_name}_{model_type}_cv{cv_folds}_wine_quality.log"

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

# 主函数
def main():
    X, y = load_data()

    # 根据输入的 model_type 获取对应模型
    model = NaiveBayesModel()
    model.set_model_type(args.model_type)

    # 创建日志记录器
    logger = create_logger(args.model, args.model_type, args.cv_folds)

    # 记录模型开始训练和超参数信息
    logger.info(f"Starting training for Naive Bayes model: {args.model_type} classifier")
    logger.info(f"Model hyperparameters: {model.hyperparameters}")

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    # 根据模型类型选择合适的预处理方法
    if args.model_type == "multinomial":
        # MultinomialNB 要求特征值为非负，使用 MinMaxScaler 确保非负
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        # 对于其他模型类型，使用 StandardScaler 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 训练模型
    model.fit(X_train, y_train)

    # 获取预测的概率分布
    y_pred_proba = model.predict_proba(X_test)

    # 计算负对数损失
    loss = log_loss(y_test, y_pred_proba)
    logger.info(f"Negative Log-Loss: {loss:.4f}")

    # 使用 zero_division=1 处理 Precision 为 NaN 的情况
    class_report = classification_report(y_test, model.predict(X_test), zero_division=1)
    logger.info(f"Classification Report:\n{class_report}")

    # 记录测试集准确率
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")

    # 记录类别分布
    logger.info(f"Class distribution:\n{pd.Series(y).value_counts()}")

    # 打印到控制台
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Negative Log-Loss: {loss:.4f}")
    print(f"Classification Report:\n{class_report}")

if __name__ == "__main__":
    main()

