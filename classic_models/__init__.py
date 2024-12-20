from .Logistic_regression import LogisticRegressionModel
from .Naive_Bayes import NaiveBayesModel

models = {
    'logistic_regression': LogisticRegressionModel,
    'naive_bayes': NaiveBayesModel
}

def find_model_using_name(model_name):
    """
    根据模型名称返回模型实例
    """
    if model_name == "naive_bayes":
        # 返回 NaiveBayesModel 的实例
        return NaiveBayesModel(model_type="bernoulli")  # 默认选择 MultinomialNB
    elif model_name in models:
        # 实例化并返回对应的模型
        return models[model_name]()
    else:
        raise ValueError(f"Model '{model_name}' not found. Available models are: {', '.join(models.keys())}")
