# 该类是一个分布式联合估计下的评估器，用于评估联合估计的性能
# 该类的输入是一个联合估计器，输出是一个评估结果
# 联合估计器的输入是来自于K+1个不同线性模型的观测数据，其中包括一个目标模型与K个辅助模型。输出是估计目标模型的参数
# 这K+1个模型使用相同的特征，但其样本的分布、噪声的分布、模型的参数都可能不同，特征之间的相关性也可能不同

import numpy as np

# 生成样本数据
# 输入参数：有效特征个数，真实回归系数，总特征数量，特征的协方差矩阵，噪声的均值，噪声的方差，样本数量
def coef_gen(coef_true, n_total, cov, noise_mean, noise_var, n_samples):
    # 生成特征
    X = np.random.multivariate_normal(np.zeros(n_total), cov, n_samples)
    # 生成噪声
    noise = np.random.normal(noise_mean, noise_var, n_samples)
    # 将真实回归系数补全至特征数量
    coef_true = np.concatenate((coef_true, np.zeros(n_total - len(coef_true))))
    # 生成标签
    y = np.dot(X, coef_true) + noise
    return X, y

class evaluator:
    def __init__(self) -> None:
        
    #独立估计器：特征间的相关性为0，噪声的分布相同，模型参数不同
    def 