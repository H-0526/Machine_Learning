import os
import logging
from datetime import datetime

# 设置日志文件夹和路径
LOG_DIR = os.path.join(os.path.dirname(__file__), "log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# 配置日志
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def log_and_run(process, *args, **kwargs):
    """
    包装函数，用于记录每个处理过程的日志
    :param process: 函数对象
    :param args: 函数的位置参数
    :param kwargs: 函数的关键字参数
    """
    process_name = process.__name__
    try:
        logging.info(f"开始运行: {process_name}")
        process(*args, **kwargs)
        logging.info(f"完成运行: {process_name}")
    except Exception as e:
        logging.error(f"运行 {process_name} 时发生错误: {e}")
        raise


# 导入f_e下的子模块
from cleaning.remove_outliers import process_concrete_data, process_combined_wine_data
from f_e.select_features import process_dataset
from f_e.standardize_data import standardize_data

if __name__ == "__main__":
    # 路径配置
    BASE_DIR = r"G:\Pycharm\Machine_Learning"
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data")

    try:
        # 1. 移除异常值
        log_and_run(process_concrete_data)
        log_and_run(process_combined_wine_data)

        # 2. 特征选择
        datasets = [
            {
                "name": "concrete",
                "correlation_file": "concrete_correlation_matrix.csv",
                "cleaned_file": "cleaned_concrete_data.csv",
                "target_variable": "Concrete compressive strength(MPa, megapascals)",
                "threshold": 0.1
            },
            {
                "name": "combined_wine",
                "correlation_file": "combined_wine_correlation_matrix.csv",
                "cleaned_file": "cleaned_combined_wine_data.csv",
                "target_variable": "quality",
                "threshold": 0.05
            }
        ]
        for dataset in datasets:
            log_and_run(
                process_dataset,
                dataset_name=dataset["name"],
                correlation_file=dataset["correlation_file"],
                cleaned_file=dataset["cleaned_file"],
                target_variable=dataset["target_variable"],
                threshold=dataset["threshold"]
            )

        # 3. 标准化处理
        standardized_datasets = [
            {
                "input_file": "selected_features_data_concrete.csv",
                "output_file": "standardized_selected_features_data_concrete.csv"
            },
            {
                "input_file": "selected_features_data_combined_wine.csv",
                "output_file": "standardized_selected_features_data_combined_wine.csv"
            }
        ]
        for dataset in standardized_datasets:
            input_path = os.path.join(PROCESSED_DATA_PATH, dataset["input_file"])
            output_path = os.path.join(PROCESSED_DATA_PATH, dataset["output_file"])
            scaler_path = os.path.join(PROCESSED_DATA_PATH, "scalers")  # 设置 scaler 保存路径
            os.makedirs(scaler_path, exist_ok=True)  # 确保路径存在
            log_and_run(standardize_data, input_path, output_path, scaler_path)

        logging.info("所有数据预处理步骤已完成！")

    except Exception as e:
        logging.error(f"数据预处理过程中发生错误: {e}")
        print(f"数据预处理失败: {e}")
