# Machine Learning Repository for Coursework

Welcome to the **Machine Learning Repository**! This project contains the implementation of machine learning tasks as outlined in the coursework document. The focus is on solving **classification** and **regression** problems using classical and advanced machine learning techniques, accompanied by exploratory data analysis and visualization.

## Repository Overview

The repository is structured into the following main sections:

1. **`/classification`**:  
   - Implements machine learning classification models for the **Wine Quality Data Set**.  
   - Includes:
     - 数据预处理（处理缺失值、异常值、标准化等）。
     - 数据探索分析（EDA），包括属性之间的相关性分析和可视化。
     - 特征工程技术，如标准化和可选的降维。
     - 实现了11种分类算法。
     - 聚类和降维结果与监督分类结果的对比分析。
     - 模型评估指标的可视化。

2. **`/regression`**:  
   - Focuses on regression analysis for the **Concrete Compressive Strength Data Set**.  
   - Includes:
     - 数据预处理和缺失数据处理。
     - 数据探索分析，研究输入特征与目标变量之间的关系。
     - 特征工程方法，例如标准化或可选的对数变换。
     - 实现8种回归算法。
     - 模型性能指标的可视化。

3. **`/reports`**:  
   - Contains detailed reports for both classification and regression tasks.  
   - Each report includes:
     - 数据描述与分析。
     - 使用的方法（预处理、建模等）。
     - 结果与讨论，包括直接从代码生成的图表。

4. **`/datasets`**:  
   - Stores the datasets used for the project:  
     - `winequality-red.csv` and `winequality-white.csv` for the classification task.  
     - `Concrete_Data.csv` for the regression task.

5. **`/scripts`**:  
   - Python scripts used for training, testing, and evaluating machine learning models.  
   - 模块化代码以便轻松复现结果。

6. **`README.md`**:  
   - This document provides an overview of the repository.

## Features

### Classification
- **Dataset**: Wine Quality Data Set (Red and White Wines).
- **Key Tasks**:
  - 执行EDA，包括属性间的相关性分析。
  - 实现监督分类算法（例如决策树、逻辑回归、支持向量机）。
  - 将聚类结果与监督分类输出进行比较。
  - 可视化模型性能指标。

### Regression
- **Dataset**: Concrete Compressive Strength Data Set.
- **Key Tasks**:
  - 执行EDA，了解输入特征与目标变量之间的关系。
  - 使用回归模型（例如线性回归、支持向量回归）进行预测。
  - 评估和可视化模型性能。

## Installation

To set up the repository, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/H-0526/Machine_learning.git
   cd Machine_learning
