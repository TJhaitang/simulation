import time
import numpy as np
import random

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


# n_packs为辅助模型数量+1（目标模型）
# 将目标模型的samples_pack置于列表第一
#-?A0指的是有效辅助模型的数量，对应算法内的L，这里暂时不考虑
def indep_eval(n_features,s,n_packs,n_samples):
    coef_true = np.zeros(n_features)
    coef_true[:s] = 0.3
    samples_packs=[]
    # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
    # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
    #生成样本
    for i in range(n_packs):
        cov = np.eye(n_features)
        delta = np.zeros(n_features)
        #delta=e*i*0.01
        # delta[:]=0.01*i
        coef=coef_true+delta
        noise_mean = 0
        noise_var = 1
        X, y = coef_gen(coef, cov, noise_mean, noise_var, n_samples)
        samples_packs.append(samples_pack(X, y))
    return samples_packs,coef_true


def t12_eval(n_features,s,n_packs,n_samples,h,L):
    coef_true = np.zeros(n_features)
    coef_true[:s] = 0.3
    samples_packs=[]
    # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
    # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
    #生成样本
    for i in range(n_packs):
        cov = np.eye(n_features)
        delta = np.zeros(n_features)
        if i==0:
            pass
        elif i<L+1:
            #delta的前10项为独立同分布的高斯随机变量
            delta[:10]=np.random.normal(0,h/100,10)
        else:
            delta[:10]=np.random.normal(0,2*s/100,10)
        coef=coef_true+delta
        noise_mean = 0
        noise_var = 1
        X, y = coef_gen(coef, cov, noise_mean, noise_var, n_samples)
        samples_packs.append(samples_pack(X, y))
    return samples_packs,coef_true

def t22_eval(n_features,s,n_packs,n_samples,h,L):
    coef_true = np.zeros(n_features)
    coef_true[:s] = 0.3
    samples_packs=[]
    # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
    # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
    #生成样本
    for i in range(n_packs):
        
        cov=None
        if i<L+1:
            #i,i为1，i,j为0.8
            cov = np.zeros((n_features,n_features))
            for j in range(n_features):
                for k in range(n_features):
                    if k==j:
                        cov[j][k]=1
                    else:
                        cov[j][k]=0.8
            
        else:
            cov = np.zeros((n_features,n_features))
            for j in range(n_features):
                for k in range(n_features):
                    if k==j:
                        cov[j][k]=1
                    elif np.abs(j-k)<2*i+1:
                        cov[j][k]=1/(i+2)
        
        delta = np.zeros(n_features)
        #在0~n_features之间随机选取h个位置，将其回归系数减0.3
        if i==0:
            pass
        elif i<L+1:
            #delta的前100项为独立同分布的高斯随机变量
            delta[:10]=np.random.normal(0,h/100,10)
        else:
            delta[:10]=np.random.normal(0,2*s/100,10)
        coef=coef_true+delta
        noise_mean = 0
        noise_var = 1
        X, y = coef_gen(coef, cov, noise_mean, noise_var, n_samples)
        samples_packs.append(samples_pack(X, y))
    #模型拟合
    return samples_packs,coef_true

def t32_eval(n_features,s,n_packs,n_samples,h,L):
    coef_true = np.zeros(n_features)
    coef_true[:s] = 0.3
    samples_packs=[]
    # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
    # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
    #生成样本
    for i in range(n_packs):
        cov=None
        if i<1:
            cov = np.eye(n_features)
        else:
            cov = np.zeros((n_features,n_features))
            for j in range(n_features):
                for k in range(n_features):
                    if k==j:
                        cov[j][k]=1
                    elif np.abs(j-k)<2*i+1:
                        cov[j][k]=1/(i+2)
        
        delta = np.zeros(n_features)
        #在0~n_features之间随机选取h个位置，将其回归系数减0.3
        if i==0:
            pass
        elif i<L+1:
            #delta的前100项为独立同分布的高斯随机变量
            delta[:10]=np.random.normal(0,h/100,10)
        else:
            delta[:10]=np.random.normal(0,2*s/100,10)
        coef=coef_true+delta
        noise_mean = 0
        noise_var = 1
        X, y = coef_gen(coef, cov, noise_mean, noise_var, n_samples)
        samples_packs.append(samples_pack(X, y))
    return samples_packs,coef_true

def t11_eval(n_features,s,n_packs,n_samples,h,L):
    coef_true = np.zeros(n_features)
    coef_true[:s] = 0.3
    samples_packs=[]
    # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
    # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
    #生成样本
    for i in range(n_packs):
        cov = np.eye(n_features)
        delta = np.zeros(n_features)
        #在0~n_features之间随机选取h个位置，将其回归系数减0.3
        if i==0:
            pass
        elif i<L+1:
            random_list=random.sample(range(n_features),h)
            delta[random_list]=-0.2
        else:
            random_list=random.sample(range(n_features),12)
            delta[random_list]=-0.5
        coef=coef_true+delta
        noise_mean = 0
        noise_var = 1
        X, y = coef_gen(coef, cov, noise_mean, noise_var, n_samples)
        samples_packs.append(samples_pack(X, y))
    return samples_packs,coef_true

def t21_eval(n_features,s,n_packs,n_samples,h,L):
    coef_true = np.zeros(n_features)
    coef_true[:s] = 0.3
    samples_packs=[]
    # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
    # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
    #生成样本
    for i in range(n_packs):
        
        cov=None
        if i<L+1:
            #i,i为1，i,j为0.8
            cov = np.zeros((n_features,n_features))
            for j in range(n_features):
                for k in range(n_features):
                    if k==j:
                        cov[j][k]=1
                    else:
                        cov[j][k]=0.8
            
        else:
            cov = np.zeros((n_features,n_features))
            for j in range(n_features):
                for k in range(n_features):
                    if k==j:
                        cov[j][k]=1
                    elif np.abs(j-k)<2*i+1:
                        cov[j][k]=1/(i+2)
        
        delta = np.zeros(n_features)
        #在0~n_features之间随机选取h个位置，将其回归系数减0.3
        if i==0:
            pass
        elif i<L+1:
            random_list=random.sample(range(n_features),h)
            delta[random_list]=-0.3
        else:
            random_list=random.sample(range(n_features),12)
            delta[random_list]=-0.5
        coef=coef_true+delta
        noise_mean = 0
        noise_var = 1
        X, y = coef_gen(coef, cov, noise_mean, noise_var, n_samples)
        samples_packs.append(samples_pack(X, y))
    return samples_packs,coef_true

def t31_eval(n_features,s,n_packs,n_samples,h,L):
    coef_true = np.zeros(n_features)
    coef_true[:s] = 0.3
    samples_packs=[]
    # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
    # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
    #生成样本
    for i in range(n_packs):
        
        cov=None
        if i<1:
            cov = np.eye(n_features)
        else:
            cov = np.zeros((n_features,n_features))
            for j in range(n_features):
                for k in range(n_features):
                    if k==j:
                        cov[j][k]=1
                    elif np.abs(j-k)<2*i+1:
                        cov[j][k]=1/(i+2)
        
        delta = np.zeros(n_features)
        #在0~n_features之间随机选取h个位置，将其回归系数减0.3
        if i==0:
            pass
        elif i<L+1:
            random_list=random.sample(range(n_features),h)
            delta[random_list]=-0.3
        else:
            random_list=random.sample(range(n_features),12)
            delta[random_list]=-0.5
        coef=coef_true+delta
        noise_mean = 0
        noise_var = 1
        X, y = coef_gen(coef, cov, noise_mean, noise_var, n_samples)
        samples_packs.append(samples_pack(X, y))
    return samples_packs,coef_true

def same_eval(n_features,s,n_packs,n_samples,h,L):
    coef_true = np.zeros(n_features)
    coef_true[:s] = 0.5
    samples_packs=[]
    for i in range(n_packs):
        cov = np.eye(n_features)
        delta = np.zeros(n_features)
        coef=coef_true+delta
        noise_mean = 0
        noise_var = 1
        X, y = coef_gen(coef, cov, noise_mean, noise_var, n_samples)
        samples_packs.append(samples_pack(X, y))
    return samples_packs,coef_true