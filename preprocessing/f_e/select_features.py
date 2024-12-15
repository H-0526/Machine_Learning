import os
import pandas as pd

# 设置路径
BASE_DIR = r"G:\Pycharm\Machine_Learning"
EDA_PATH = os.path.join(BASE_DIR, "eda", "visualizations")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data")
OUTPUT_PATH = PROCESSED_DATA_PATH
os.makedirs(OUTPUT_PATH, exist_ok=True)

def select_features(correlation_matrix_path, target_variable, threshold=0.1):
    """
    根据相关性选择特征
    """
    # 加载相关性矩阵
    correlation_matrix = pd.read_csv(correlation_matrix_path, index_col=0)

    # 去除列名和索引中的多余空格或特殊字符
    correlation_matrix.columns = correlation_matrix.columns.str.strip().str.replace('"', '').str.replace("  ", " ")
    correlation_matrix.index = correlation_matrix.index.str.strip().str.replace('"', '').str.replace("  ", " ")

    # 打印列名和索引以检查
    print("相关性矩阵的列名:", correlation_matrix.columns.tolist())
    print("相关性矩阵的索引:", correlation_matrix.index.tolist())

    # 确保目标变量存在于相关性矩阵中
    if target_variable not in correlation_matrix.columns:
        print(f"目标变量 '{target_variable}' 不在相关性矩阵中。")
        print("当前相关性矩阵的列名:", correlation_matrix.columns.tolist())
        raise ValueError(f"目标变量 '{target_variable}' 不在相关性矩阵中。")

    # 筛选与目标变量相关性大于阈值的特征
    target_correlation = correlation_matrix[target_variable]
    selected_features = target_correlation[abs(target_correlation) >= threshold].index.tolist()

    # 排除目标变量本身
    if target_variable in selected_features:
        selected_features.remove(target_variable)

    print(f"与目标变量 '{target_variable}' 相关性 >= {threshold} 的特征: {selected_features}")
    return selected_features

def save_selected_features(input_data_path, selected_features, target_variable, output_path, dataset_name):
    # 加载处理后的数据集
    processed_data = pd.read_csv(input_data_path)

    # 去除列名中的多余空格或特殊字符
    processed_data.columns = processed_data.columns.str.strip().str.replace('"', '').str.replace("  ", " ")

    # 提取目标特征列和目标变量列
    columns_to_save = selected_features + [target_variable]

    # 检查列名是否存在
    missing_columns = [col for col in columns_to_save if col not in processed_data.columns]
    if missing_columns:
        raise ValueError(f"以下列未找到: {missing_columns}")

    selected_data = processed_data[columns_to_save]

    # 保存处理后的数据
    output_file = os.path.join(output_path, f"selected_features_data_{dataset_name}.csv")
    selected_data.to_csv(output_file, index=False)
    print(f"已保存选择特征后的数据至 {output_file}")

def process_dataset(dataset_name, correlation_file, cleaned_file, target_variable, threshold):
    """
    处理单个数据集的特征选择
    :param dataset_name: str, 数据集名称（如 combined_wine, concrete）
    :param correlation_file: str, 相关性文件名
    :param cleaned_file: str, 清理后的数据集文件名
    :param target_variable: str, 目标变量名称
    :param threshold: float, 相关性阈值
    """
    correlation_matrix_path = os.path.join(EDA_PATH, correlation_file)
    input_data_path = os.path.join(PROCESSED_DATA_PATH, cleaned_file)

    # 选择特征
    selected_features = select_features(correlation_matrix_path, target_variable, threshold)

    # 保存选择后的数据
    save_selected_features(input_data_path, selected_features, target_variable, OUTPUT_PATH, dataset_name)

if __name__ == "__main__":
    # 定义数据集相关信息
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
        try:
            print(f"\n处理数据集: {dataset['name']}")
            process_dataset(
                dataset_name=dataset["name"],
                correlation_file=dataset["correlation_file"],
                cleaned_file=dataset["cleaned_file"],
                target_variable=dataset["target_variable"],
                threshold=dataset["threshold"]
            )
        except Exception as e:
            print(f"处理数据集 {dataset['name']} 时出错: {e}")
