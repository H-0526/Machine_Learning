import os
import importlib

# 确保运行时当前工作目录正确
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# 确保文件路径存在
visualizations_path = "visualizations"
os.makedirs(visualizations_path, exist_ok=True)

def main():
    """
    运行 EDA 概览脚本
    """
    # 打印任务信息
    print("开始执行 EDA 任务...\n")

    try:
        # 调用相关性分析脚本
        print("调用 correlation_analysis.py 进行相关性分析...")
        correlation_analysis = importlib.import_module("correlation_analysis")
        correlation_analysis.analyze_correlation_concrete()
        correlation_analysis.analyze_correlation_wine()
    except Exception as e:
        print(f"执行相关性分析时出错: {e}")

    try:
        # 调用分布分析脚本
        print("\n调用 distribution_analysis.py 进行分布分析...")
        distribution_analysis = importlib.import_module("distribution_analysis")
        distribution_analysis.analyze_distribution_concrete()
        distribution_analysis.analyze_distribution_wine()
    except Exception as e:
        print(f"执行分布分析时出错: {e}")

    print("\n所有 EDA 任务已完成！可视化结果保存在 visualizations 文件夹中。")

if __name__ == "__main__":
    main()
