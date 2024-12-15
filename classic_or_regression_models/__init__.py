from .DeepRF import DeepRFModel
from .ETR import ETRModel
from .GBDT import GBDTModel
from .LightGBM import LightGBM
from .RF import RFModel
from .SVM import SVMModel
from .XGBoost import XGBoostModel

models = {
    'deep_rf': DeepRFModel,
    'etr': ETRModel,
    'gbdt': GBDTModel,
    'lightgbm': LightGBM,
    'rf': RFModel,
    'svm': SVMModel,
    'xgboost': XGBoostModel
}

def find_model_using_name(model_name, task="classification"):
    """
    根据模型名称和任务类型返回模型实例

    :param model_name: 模型名称
    :param task: 任务类型 ("classification" 或 "regression")
    :return: 相应模型的实例
    """
    if model_name == "deep_rf":
        # 返回 DeepRFModel 实例
        return DeepRFModel(task=task)
    elif model_name == "etr":
        # 返回 ETRModel 实例
        return ETRModel(task=task)
    elif model_name == "gbdt":
        # 返回 GBDTModel 实例
        return GBDTModel(task=task)
    elif model_name == "lightgbm":
        # 返回 LightGBM 实例
        return LightGBM(task=task)
    elif model_name == "rf":
        # 返回 RFModel 实例
        return RFModel(task=task)
    elif model_name == "svm":
        # 返回 SVMModel 实例
        return SVMModel(task=task)
    elif model_name == "xgboost":
        # 返回 XGBoostModel 实例
        return XGBoostModel(task=task)
    elif model_name in models:
        # 返回指定的模型实例
        return models[model_name](task=task)
    else:
        raise ValueError(f"Model '{model_name}' not found. Available models are: {', '.join(models.keys())}")
