import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置结果存储路径
visualizations_path = "visualizations"

# 确保可视化结果文件夹存在
os.makedirs(visualizations_path, exist_ok=True)


def analyze_distribution_concrete():
    """
    对混凝土抗压强度数据集进行分布分析
    """
    concrete_data_path = r"G:\Pycharm\Machine_Learning\data\raw_data\concrete+compressive+strength\Concrete_Data.xls"
    # 加载数据
    concrete_df = pd.read_excel(concrete_data_path)

    # 遍历所有特征，绘制直方图和箱线图
    for column in concrete_df.columns:
        # 绘制直方图
        plt.figure(figsize=(8, 6))
        sns.histplot(concrete_df[column], kde=True, bins=30, color="blue")
        plt.title(f"Histogram - {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, f"concrete_histogram_{column}.png"))
        plt.close()

        # 绘制箱线图
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=concrete_df[column], color="green")
        plt.title(f"Boxplot - {column}")
        plt.xlabel(column)
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, f"concrete_boxplot_{column}.png"))
        plt.close()

    print("混凝土抗压强度数据集分布分析完成！")


def analyze_distribution_wine():
    """
    对红葡萄酒和白葡萄酒质量数据集进行分布分析
    """
    wine_data_path_red = r"G:\Pycharm\Machine_Learning\data\raw_data\wine+quality\winequality-red.csv"
    wine_data_path_white = r"G:\Pycharm\Machine_Learning\data\raw_data\wine+quality\winequality-white.csv"

    # 加载红葡萄酒和白葡萄酒数据
    red_df = pd.read_csv(wine_data_path_red, sep=";")
    white_df = pd.read_csv(wine_data_path_white, sep=";")

    # 分析红葡萄酒数据
    for column in red_df.columns:
        # 绘制直方图
        plt.figure(figsize=(8, 6))
        sns.histplot(red_df[column], kde=True, bins=30, color="red")
        plt.title(f"Histogram - Red Wine - {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, f"red_wine_histogram_{column}.png"))
        plt.close()

        # 绘制箱线图
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=red_df[column], color="orange")
        plt.title(f"Boxplot - Red Wine - {column}")
        plt.xlabel(column)
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, f"red_wine_boxplot_{column}.png"))
        plt.close()

    print("红葡萄酒质量数据集分布分析完成！")

    # 分析白葡萄酒数据
    for column in white_df.columns:
        # 绘制直方图
        plt.figure(figsize=(8, 6))
        sns.histplot(white_df[column], kde=True, bins=30, color="purple")
        plt.title(f"Histogram - White Wine - {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, f"white_wine_histogram_{column}.png"))
        plt.close()

        # 绘制箱线图
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=white_df[column], color="yellow")
        plt.title(f"Boxplot - White Wine - {column}")
        plt.xlabel(column)
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, f"white_wine_boxplot_{column}.png"))
        plt.close()

    print("白葡萄酒质量数据集分布分析完成！")


def analyze_distribution_combined_wine():
    """
    对红白葡萄酒组合数据集进行分布分析
    """
    wine_data_path_red = r"G:\Pycharm\Machine_Learning\data\raw_data\wine+quality\winequality-red.csv"
    wine_data_path_white = r"G:\Pycharm\Machine_Learning\data\raw_data\wine+quality\winequality-white.csv"

    # 加载红葡萄酒和白葡萄酒数据
    red_df = pd.read_csv(wine_data_path_red, sep=";")
    white_df = pd.read_csv(wine_data_path_white, sep=";")

    # 合并数据集
    combined_df = pd.concat([red_df, white_df], axis=0)

    # 分析组合数据
    for column in combined_df.columns:
        # 绘制直方图
        plt.figure(figsize=(8, 6))
        sns.histplot(combined_df[column], kde=True, bins=30, color="cyan")
        plt.title(f"Histogram - Combined Wine - {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, f"combined_wine_histogram_{column}.png"))
        plt.close()

        # 绘制箱线图
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=combined_df[column], color="pink")
        plt.title(f"Boxplot - Combined Wine - {column}")
        plt.xlabel(column)
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_path, f"combined_wine_boxplot_{column}.png"))
        plt.close()

    print("红白葡萄酒组合数据集分布分析完成！")


if __name__ == "__main__":
    # 分别分析数据集
    analyze_distribution_concrete()
    analyze_distribution_wine()
    analyze_distribution_combined_wine()
    print("所有分布分析任务完成！")
