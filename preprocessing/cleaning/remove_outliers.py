import os
import pandas as pd

# 设置路径
processed_data_path = r"G:\Pycharm\Machine_Learning\data\processed_data"
os.makedirs(processed_data_path, exist_ok=True)

def remove_outliers_iqr(df, column):
    """
    使用 IQR 方法检测并移除异常值
    :param df: 数据框
    :param column: 需要处理的列名
    :return: 移除异常值后的数据框
    """
    Q1 = df[column].quantile(0.25)  # 第一四分位数
    Q3 = df[column].quantile(0.75)  # 第三四分位数
    IQR = Q3 - Q1  # 四分位距
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 过滤异常值
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

def process_concrete_data():
    """
    对混凝土抗压强度数据集进行异常值处理
    """
    concrete_data_path = r"G:\Pycharm\Machine_Learning\data\raw_data\concrete+compressive+strength\Concrete_Data.xls"
    df = pd.read_excel(concrete_data_path)

    print("处理混凝土数据集异常值...")
    # 对每个列应用异常值移除
    for column in df.columns:
        before_count = len(df)
        df = remove_outliers_iqr(df, column)
        after_count = len(df)
        print(f"列 {column}：移除 {before_count - after_count} 个异常值")

    # 保存处理后的数据
    output_path = os.path.join(processed_data_path, "cleaned_concrete_data.csv")
    df.to_csv(output_path, index=False)
    print(f"清理后的混凝土数据集已保存至 {output_path}")

def process_wine_data():
    """
    对红葡萄酒和白葡萄酒质量数据集进行异常值处理
    """
    wine_data_path_red = r"G:\Pycharm\Machine_Learning\data\raw_data\wine+quality\winequality-red.csv"
    wine_data_path_white = r"G:\Pycharm\Machine_Learning\data\raw_data\wine+quality\winequality-white.csv"

    # 加载红葡萄酒和白葡萄酒数据
    red_df = pd.read_csv(wine_data_path_red, sep=";")
    white_df = pd.read_csv(wine_data_path_white, sep=";")

    print("处理红葡萄酒数据集异常值...")
    for column in red_df.columns:
        before_count = len(red_df)
        red_df = remove_outliers_iqr(red_df, column)
        after_count = len(red_df)
        print(f"列 {column}：移除 {before_count - after_count} 个异常值")

    print("处理白葡萄酒数据集异常值...")
    for column in white_df.columns:
        before_count = len(white_df)
        white_df = remove_outliers_iqr(white_df, column)
        after_count = len(white_df)
        print(f"列 {column}：移除 {before_count - after_count} 个异常值")

    # 保存处理后的数据
    red_output_path = os.path.join(processed_data_path, "cleaned_red_wine_data.csv")
    white_output_path = os.path.join(processed_data_path, "cleaned_white_wine_data.csv")
    red_df.to_csv(red_output_path, index=False)
    white_df.to_csv(white_output_path, index=False)
    print(f"清理后的红葡萄酒数据集已保存至 {red_output_path}")
    print(f"清理后的白葡萄酒数据集已保存至 {white_output_path}")

def process_combined_wine_data():
    """
    对红白葡萄酒组合数据集进行异常值处理
    """
    wine_data_path_red = r"G:\Pycharm\Machine_Learning\data\raw_data\wine+quality\winequality-red.csv"
    wine_data_path_white = r"G:\Pycharm\Machine_Learning\data\raw_data\wine+quality\winequality-white.csv"

    # 加载红葡萄酒和白葡萄酒数据
    red_df = pd.read_csv(wine_data_path_red, sep=";")
    white_df = pd.read_csv(wine_data_path_white, sep=";")

    # 合并数据集
    combined_df = pd.concat([red_df, white_df], axis=0)

    print("处理红白葡萄酒组合数据集异常值...")
    for column in combined_df.columns:
        before_count = len(combined_df)
        combined_df = remove_outliers_iqr(combined_df, column)
        after_count = len(combined_df)
        print(f"列 {column}：移除 {before_count - after_count} 个异常值")

    # 保存处理后的数据
    combined_output_path = os.path.join(processed_data_path, "cleaned_combined_wine_data.csv")
    combined_df.to_csv(combined_output_path, index=False)
    print(f"清理后的红白葡萄酒组合数据集已保存至 {combined_output_path}")

if __name__ == "__main__":
    process_concrete_data()
    process_wine_data()
    process_combined_wine_data()
    print("所有数据集的异常值处理已完成！")
