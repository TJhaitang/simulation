# 该类是一个分布式联合估计下的评估器，用于评估联合估计的性能
# 该类的输入是一个联合估计器，输出是一个评估结果
# 联合估计器的输入是来自于K+1个不同线性模型的观测数据，其中包括一个目标模型与K个辅助模型。输出是估计目标模型的参数
# 这K+1个模型使用相同的特征，但其样本的分布、噪声的分布、模型的参数都可能不同，特征之间的相关性也可能不同

import numpy as np

# 生成样本数据
#   
def coef_gen(coef, cov, noise_mean, noise_var, n_samples):
    # 生成特征
    X = np.random.multivariate_normal(np.zeros(len(coef)), cov, n_samples)
    # 生成噪声
    noise = np.random.normal(noise_mean, noise_var, n_samples)
    # 生成标签
    y = np.dot(X, coef) + noise
    return X, y

# 考虑后期以该类模拟不同数据的分布，对取数据加以时间限制等
class samples_pack:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def get_n_fretures(self):
        return len(self.X[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    def getX(self):
        return self.X
    def getY(self):
        return self.y

class evaluator:
    def __init__(self) -> None:
        pass
        
    # 独立估计器：特征间的相关性为0，噪声的分布相同，模型参数不同
    # 暂时不考虑模型样本的数量均衡性，默认全部相同
    
    def eval(self, estiminator, sample_packs):
        estiminator.fit(sample_packs)
        return estiminator.get_params()
    
    # n_packs为辅助模型数量+1（目标模型）
    # 将目标模型的samples_pack置于列表第一
    #-?A0指的是有效辅助模型的数量，对应算法内的L，这里暂时不考虑
    def indep_eval(self, estiminator, n_packs, n_samples,n_features,s,A0):
        samples_packs=[]
        # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
        # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
        coef_true = np.zeros(n_features)
        coef_true[:s] = 0.3
        #生成样本
        for i in range(n_packs):
            cov = np.eye(n_features)
            noise_mean = 0
            noise_var = 0.1
            X, y = coef_gen(coef_true, cov, noise_mean, noise_var, n_samples)
            samples_packs.append(samples_pack(X, y))
        #模型拟合
        pre_params = self.eval(estiminator, samples_packs)
        #计算coef_true与pre_params的SSE
        SSE = np.sum((coef_true - pre_params)**2)
        return SSE
        

        

        