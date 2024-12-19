import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# 设置文件路径
BASE_DIR = r"G:\Pycharm\Machine_Learning\data\processed_data"
COMBINED_WINE_FILE = os.path.join(BASE_DIR, "selected_features_data_combined_wine.csv")
CONCRETE_FILE = os.path.join(BASE_DIR, "selected_features_data_concrete.csv")

# 保存标准化后的数据路径
STANDARDIZED_WINE_FILE = os.path.join(BASE_DIR, "standardized_selected_features_data_combined_wine.csv")
STANDARDIZED_CONCRETE_FILE = os.path.join(BASE_DIR, "standardized_selected_features_data_concrete.csv")

# 保存 scaler 的路径
SCALER_BASE_PATH = r"G:\Pycharm\Machine_Learning\data\processed_data\scalers"
os.makedirs(SCALER_BASE_PATH, exist_ok=True)

def standardize_data(input_path, output_path, scaler_path):
    """
    对数据集进行标准化并保存
    :param input_path: str, 输入数据文件路径
    :param output_path: str, 标准化后数据保存路径
    :param scaler_path: str, 标准化器保存路径
    """
    print(f"正在处理数据集: {input_path}")

    # 加载数据
    data = pd.read_csv(input_path)

    # 获取目标变量的列名
    target_column = data.columns[-1]

    # 提取特征和目标变量
    features = data.drop(columns=[target_column])
    target = data[target_column]

    # 初始化标准化器
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # 标准化特征和目标变量
    standardized_features = feature_scaler.fit_transform(features)
    standardized_target = target_scaler.fit_transform(target.values.reshape(-1, 1))

    # 转换为 DataFrame 并重新合并目标变量
    standardized_data = pd.DataFrame(standardized_features, columns=features.columns)
    standardized_data[target_column] = standardized_target

    # 保存标准化后的数据
    standardized_data.to_csv(output_path, index=False)
    print(f"标准化后的数据已保存至: {output_path}")

    # 保存 scalers 以供反向变换
    feature_scaler_file = os.path.join(scaler_path, f"feature_scaler_{os.path.basename(input_path).split('.')[0]}.joblib")
    target_scaler_file = os.path.join(scaler_path, f"target_scaler_{os.path.basename(input_path).split('.')[0]}.joblib")
    joblib.dump(feature_scaler, feature_scaler_file)
    joblib.dump(target_scaler, target_scaler_file)
    print(f"特征和目标变量的标准化器已保存至: {scaler_path}")


if __name__ == "__main__":
    # 对两个数据集进行标准化
    try:
        standardize_data(COMBINED_WINE_FILE, STANDARDIZED_WINE_FILE, SCALER_BASE_PATH)
        standardize_data(CONCRETE_FILE, STANDARDIZED_CONCRETE_FILE, SCALER_BASE_PATH)
        print("所有数据集已成功完成标准化处理！")
    except Exception as e:
        print(f"数据处理时发生错误: {e}")
