from .Cubist import CubistModel

# 模型映射字典
models = {
    'cubist': CubistModel  # 唯一的模型是 CubistModel
}


def find_model_using_name(model_name):
    """
    根据模型名称返回模型实例

    :param model_name: 模型名称
    :return: 相应模型的实例
    """
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available models are: {', '.join(models.keys())}")

    # 返回指定的模型实例
    return models[model_name]()
