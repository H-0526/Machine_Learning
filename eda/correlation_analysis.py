import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置结果存储路径
visualizations_path = "visualizations"

# 确保可视化结果文件夹存在
os.makedirs(visualizations_path, exist_ok=True)


def analyze_correlation_concrete():
    """
    对混凝土抗压强度数据集进行相关性分析
    """
    concrete_data_path = r"G:\Pycharm\Machine_Learning\data\raw_data\concrete+compressive+strength\Concrete_Data.xls"
    # 加载数据
    concrete_df = pd.read_excel(concrete_data_path)

    # 计算相关性矩阵
    correlation_matrix = concrete_df.corr()

    # 保存相关性矩阵为CSV
    correlation_matrix.to_csv(os.path.join(visualizations_path, "concrete_correlation_matrix.csv"))

    # 可视化相关性矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix - Concrete Compressive Strength")
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_path, "concrete_correlation_matrix.png"))
    plt.close()
    print("混凝土抗压强度数据集相关性分析完成！")


def analyze_correlation_wine():
    """
    对红葡萄酒和白葡萄酒质量数据集进行相关性分析
    """
    wine_data_path_red = r"G:\Pycharm\Machine_Learning\data\raw_data\wine+quality\winequality-red.csv"
    wine_data_path_white = r"G:\Pycharm\Machine_Learning\data\raw_data\wine+quality\winequality-white.csv"

    # 加载红葡萄酒和白葡萄酒数据
    red_df = pd.read_csv(wine_data_path_red, sep=";")
    white_df = pd.read_csv(wine_data_path_white, sep=";")

    # 计算红葡萄酒相关性矩阵
    red_corr_matrix = red_df.corr()
    red_corr_matrix.to_csv(os.path.join(visualizations_path, "red_wine_correlation_matrix.csv"))
    plt.figure(figsize=(10, 8))
    sns.heatmap(red_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix - Red Wine Quality")
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_path, "red_wine_correlation_matrix.png"))
    plt.close()
    print("红葡萄酒质量数据集相关性分析完成！")

    # 计算白葡萄酒相关性矩阵
    white_corr_matrix = white_df.corr()
    white_corr_matrix.to_csv(os.path.join(visualizations_path, "white_wine_correlation_matrix.csv"))
    plt.figure(figsize=(10, 8))
    sns.heatmap(white_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix - White Wine Quality")
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_path, "white_wine_correlation_matrix.png"))
    plt.close()
    print("白葡萄酒质量数据集相关性分析完成！")


def analyze_correlation_combined_wine():
    """
    对红白葡萄酒组合数据集进行相关性分析
    """
    wine_data_path_red = r"G:\Pycharm\Machine_Learning\data\raw_data\wine+quality\winequality-red.csv"
    wine_data_path_white = r"G:\Pycharm\Machine_Learning\data\raw_data\wine+quality\winequality-white.csv"

    # 加载红葡萄酒和白葡萄酒数据
    red_df = pd.read_csv(wine_data_path_red, sep=";")
    white_df = pd.read_csv(wine_data_path_white, sep=";")

    # 合并数据集
    combined_df = pd.concat([red_df, white_df], axis=0)

    # 计算相关性矩阵
    combined_corr_matrix = combined_df.corr()
    combined_corr_matrix.to_csv(os.path.join(visualizations_path, "combined_wine_correlation_matrix.csv"))
    plt.figure(figsize=(10, 8))
    sns.heatmap(combined_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix - Combined Red and White Wine Quality")
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_path, "combined_wine_correlation_matrix.png"))
    plt.close()
    print("红白葡萄酒组合数据集相关性分析完成！")


if __name__ == "__main__":
    # 分别分析数据集
    analyze_correlation_concrete()
    analyze_correlation_wine()
    analyze_correlation_combined_wine()
    print("所有相关性分析任务完成！")
