# 机器学习课设作业

Welcome to the **Machine Learning Project Repository**! This project is designed around the coursework requirements, focusing on solving **classification** and **regression** problems. By integrating classical and modern machine learning techniques with in-depth exploratory data analysis (EDA) and visualization, we aim to comprehensively implement machine learning tasks. From data analysis and feature engineering to model implementation and evaluation, we employ various algorithms such as logistic regression, support vector machines, and gradient boosting trees to process real-world datasets. Additionally, we explore clustering and dimensionality reduction techniques to analyze the differences between unsupervised and supervised learning outcomes. Through this project, you will experience the complete workflow from data to models, enhancing your understanding and practical skills in machine learning. Let’s delve into the fascinating world of machine learning and unlock the unlimited potential hidden in data!

![机器学习作业工作流程图 drawio](https://github.com/user-attachments/assets/023090b6-0e18-4dd0-bcbc-9b85a8b9af67)

## 分类
**1、数据集：Wine Quality Data Set**
这个数据集包含了不同葡萄酒的质量评分和其他物理化学性质，如固定酸度、挥发性酸度、柠檬酸等。总共有12个特征，质量评分作为目标变量，可以被视为一个多分类任务。红葡萄酒数据集有1599个样本，白葡萄酒数据集有4898个样本。
链接：[Wine Quality - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)

**2、设计目标**
1)	对数据集进行深入理解和介绍。
2)	进行数据探索，如进行属性间的相关性分析，如有缺失、离群点之类的，进行相应处理。
3)	特征工程，包括标准化、TF-IDF（非必须）,log(x+1)（非必须）等,如果有必要进行降维
4)	应用经典的机器学习分类算法（如决策树、逻辑回归、支持向量机、朴素贝叶斯、集成算法和BP神经网络等，不少于3种）对数据进行分类，。
5)	在去掉标签栏后，对这份数据进行降维和聚类。可以先降维后再聚类，也可以各自进行。将聚类的结果与监督分类的结果进行对比和分析。
6)	评估和比较不同分类算法，尽量将过程和结果可视化。

## 回归
**1、数据集：Concrete Compressive Strength Data Set**
描述: 该数据集包含水泥强度与组成材料之间的关系。它有8个特征（包括水泥、炉渣、飞灰、水、超细粉、粗骨料、细骨料和龄期）和一个目标变量（混凝土抗压强度）。样本量: 1030
链接: [Concrete Compressive Strength - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)

**2、设计目标**
1)	对数据集进行深入理解和介绍。
2)	进行数据探索，如进行属性间的相关性分析，如有缺失、离群点之类的，进行相应处理。
3)	特征工程，包括标准化、TF-IDF（非必须）,log(x+1)（非必须）等,如果有必要进行降维
4)	应用经典的机器学习回归算法（如线性回归、CART树、集成学习、神经网络、支持向量回归等，不少于2种）对数据预测。
5)	评估和比较不同算法，尽量将过程和结果可视化。

### Classification
- **Dataset**: Wine Quality Data Set (Red and White Wines).
- **Key Tasks**:
  - 数据探索：/eda，包括目标变量分析；变量统计汇总；变量相关性分析；由于数据集没有缺失值，因此只做了异常值检测
  - 数据预处理：/preprocessing, 包括检测并剔除异常值；基于相关性过滤的特征选择；特征变量进行数据标准化
  - 模型实验：实现监督分类算法（逻辑斯蒂回归、朴素贝叶斯、深度随机森林、极端随机树、梯度提升决策树、轻量级梯度提升机、随机森林、支持向量机、极限梯度提升）
  - 将聚类结果与监督分类输出进行比较:/Dimensionality_Reduction_and_Clustering_wine_quality
  - 统计模型性能指标：/results_preprocessed

### Regression
- **Dataset**: Concrete Compressive Strength Data Set.
- **Key Tasks**:
  - 数据探索：/eda，包括目标变量分析；变量统计汇总；变量相关性分析；由于数据集没有缺失值，因此只做了异常值检测
  - 数据预处理：/preprocessing, 包括检测并剔除异常值；基于相关性过滤的特征选择；特征变量进行数据标准化
  - 模型实验：实现监督回归算法（规则线性回归、深度随机森林、极端随机树、梯度提升决策树、轻量级梯度提升机、随机森林、支持向量机、极限梯度提升）
  - 统计模型性能指标：/results_preprocessed

## Installation

To set up the repository, follow these steps:

Clone this repository:
   ```bash
   git clone https://github.com/H-0526/Machine_Learning.git
   cd Machine_learning
