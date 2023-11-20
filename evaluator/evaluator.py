# 该类是一个分布式联合估计下的评估器，用于评估联合估计的性能
# 该类的输入是一个联合估计器，输出是一个评估结果
# 联合估计器的输入是来自于K+1个不同线性模型的观测数据，其中包括一个目标模型与K个辅助模型。输出是估计目标模型的参数
# 这K+1个模型使用相同的特征，但其样本的分布、噪声的分布、模型的参数都可能不同，特征之间的相关性也可能不同

import time
import numpy as np
import random

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
    def __init__(self,n_features, n_packs, n_samples, s, L,repeat_times=1) -> None:
        self.n_packs=n_packs
        self.n_samples=n_samples
        self.n_features=n_features
        self.s=s
        self.L=L
        self.samples_packs=None
        self.method=None
        self.repeat_times=repeat_times
        
    # 独立估计器：特征间的相关性为0，噪声的分布相同，模型参数不同
    # 暂时不考虑模型样本的数量均衡性，默认全部相同
    
    def eval(self, estiminator, sample_packs,coef_true):
        #计时
        SSE_list=[]
        time_list=[]
        for i in range(self.repeat_times):
            start=time.time()
            estiminator.fit(sample_packs,self.s,self.L)
            end=time.time()
            time_list.append(end-start)
            SSE=np.sum((coef_true-estiminator.get_params())**2)
            SSE_list.append(SSE)
        SSE_mean=np.mean(SSE_list)
        time_mean=np.mean(time_list)
        return coef_true,SSE_mean,time_mean
    
    # n_packs为辅助模型数量+1（目标模型）
    # 将目标模型的samples_pack置于列表第一
    #-?A0指的是有效辅助模型的数量，对应算法内的L，这里暂时不考虑
    def indep_eval(self, estiminator,refresh=False):
        coef_true = np.zeros(self.n_features)
        coef_true[:self.s] = 0.3
        if refresh or self.samples_packs==None or self.method!='indep':
            samples_packs=[]
            # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
            # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
            #生成样本
            for i in range(self.n_packs):
                cov = np.eye(self.n_features)
                delta = np.zeros(self.n_features)
                #delta=e*i*0.01
                # delta[:]=0.01*i
                coef=coef_true+delta
                noise_mean = 0
                noise_var = 1
                X, y = coef_gen(coef, cov, noise_mean, noise_var, self.n_samples)
                samples_packs.append(samples_pack(X, y))
            self.samples_packs=samples_packs
            self.method='indep'
        #模型拟合
        return self.eval(estiminator, self.samples_packs,coef_true)

    
    def t12_eval(self,estiminator,h,refresh=False):
        coef_true = np.zeros(self.n_features)
        coef_true[:self.s] = 0.3
        if refresh or self.samples_packs==None or self.method!='t12':
            samples_packs=[]
            # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
            # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
            #生成样本
            for i in range(self.n_packs):
                cov = np.eye(self.n_features)
                delta = np.zeros(self.n_features)
                if i==0:
                    pass
                elif i<self.L+1:
                    #delta的前100项为独立同分布的高斯随机变量
                    delta[:100]=np.random.normal(0,h/100,100)
                else:
                    delta[:100]=np.random.normal(0,2*self.s/100,100)
                coef=coef_true+delta
                noise_mean = 0
                noise_var = 1
                X, y = coef_gen(coef, cov, noise_mean, noise_var, self.n_samples)
                samples_packs.append(samples_pack(X, y))
            self.samples_packs=samples_packs
            self.method='t12'
        #模型拟合
        return self.eval(estiminator, self.samples_packs,coef_true)
    
    def t22_eval(self,estiminator,h,refresh=False):
        coef_true = np.zeros(self.n_features)
        coef_true[:self.s] = 0.3
        if refresh or self.samples_packs==None or self.method!='t22':
            samples_packs=[]
            # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
            # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
            #生成样本
            for i in range(self.n_packs):
                
                cov=None
                if i<self.L+1:
                    #i,i为1，i,j为0.8
                    cov = np.zeros((self.n_features,self.n_features))
                    for j in range(self.n_features):
                        for k in range(self.n_features):
                            if k==j:
                                cov[j][k]=1
                            else:
                                cov[j][k]=0.8
                    
                else:
                    cov = np.zeros((self.n_features,self.n_features))
                    for j in range(self.n_features):
                        for k in range(self.n_features):
                            if k==j:
                                cov[j][k]=1
                            elif np.abs(j-k)<2*i+1:
                                cov[j][k]=1/(i+2)
                
                delta = np.zeros(self.n_features)
                #在0~n_features之间随机选取h个位置，将其回归系数减0.3
                if i==0:
                    pass
                elif i<self.L+1:
                    #delta的前100项为独立同分布的高斯随机变量
                    delta[:100]=np.random.normal(0,h/100,100)
                else:
                    delta[:100]=np.random.normal(0,2*self.s/100,100)
                coef=coef_true+delta
                noise_mean = 0
                noise_var = 1
                X, y = coef_gen(coef, cov, noise_mean, noise_var, self.n_samples)
                samples_packs.append(samples_pack(X, y))
            self.samples_packs=samples_packs
            self.method='t22'
        #模型拟合
        return self.eval(estiminator, self.samples_packs,coef_true)
    
    def t32_eval(self,estiminator,h,refresh=False):
        coef_true = np.zeros(self.n_features)
        coef_true[:self.s] = 0.3
        if refresh or self.samples_packs==None or self.method!='t32':
            samples_packs=[]
            # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
            # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
            #生成样本
            for i in range(self.n_packs):
                
                cov=None
                if i<1:
                    cov = np.eye(self.n_features)
                else:
                    cov = np.zeros((self.n_features,self.n_features))
                    for j in range(self.n_features):
                        for k in range(self.n_features):
                            if k==j:
                                cov[j][k]=1
                            elif np.abs(j-k)<2*i+1:
                                cov[j][k]=1/(i+2)
                
                delta = np.zeros(self.n_features)
                #在0~n_features之间随机选取h个位置，将其回归系数减0.3
                if i==0:
                    pass
                elif i<self.L+1:
                    #delta的前100项为独立同分布的高斯随机变量
                    delta[:100]=np.random.normal(0,h/100,100)
                else:
                    delta[:100]=np.random.normal(0,2*self.s/100,100)
                coef=coef_true+delta
                noise_mean = 0
                noise_var = 1
                X, y = coef_gen(coef, cov, noise_mean, noise_var, self.n_samples)
                samples_packs.append(samples_pack(X, y))
            self.samples_packs=samples_packs
            self.method='t32'
        #模型拟合
        return self.eval(estiminator, self.samples_packs,coef_true)
    
    def t11_eval(self,estiminator,h,refresh=False):
        coef_true = np.zeros(self.n_features)
        coef_true[:self.s] = 0.3
        if refresh or self.samples_packs==None or self.method!='t11':
            samples_packs=[]
            # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
            # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
            #生成样本
            for i in range(self.n_packs):
                cov = np.eye(self.n_features)
                delta = np.zeros(self.n_features)
                #在0~n_features之间随机选取h个位置，将其回归系数减0.3
                if i==0:
                    pass
                elif i<self.L+1:
                    random_list=random.sample(range(self.n_features),h)
                    delta[random_list]=-0.3
                else:
                    random_list=random.sample(range(self.n_features),int(self.s/2))
                    delta[random_list]=-0.5
                coef=coef_true+delta
                noise_mean = 0
                noise_var = 1
                X, y = coef_gen(coef, cov, noise_mean, noise_var, self.n_samples)
                samples_packs.append(samples_pack(X, y))
            self.samples_packs=samples_packs
            self.method='t11'
        #模型拟合
        return self.eval(estiminator, self.samples_packs,coef_true)
    
    def t21_eval(self,estiminator,h,refresh=False):
        coef_true = np.zeros(self.n_features)
        coef_true[:self.s] = 0.3
        if refresh or self.samples_packs==None or self.method!='t21':
            samples_packs=[]
            # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
            # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
            #生成样本
            for i in range(self.n_packs):
                
                cov=None
                if i<self.L+1:
                    #i,i为1，i,j为0.8
                    cov = np.zeros((self.n_features,self.n_features))
                    for j in range(self.n_features):
                        for k in range(self.n_features):
                            if k==j:
                                cov[j][k]=1
                            else:
                                cov[j][k]=0.8
                    
                else:
                    cov = np.zeros((self.n_features,self.n_features))
                    for j in range(self.n_features):
                        for k in range(self.n_features):
                            if k==j:
                                cov[j][k]=1
                            elif np.abs(j-k)<2*i+1:
                                cov[j][k]=1/(i+2)
                
                delta = np.zeros(self.n_features)
                #在0~n_features之间随机选取h个位置，将其回归系数减0.3
                if i==0:
                    pass
                elif i<self.L+1:
                    random_list=random.sample(range(self.n_features),h)
                    delta[random_list]=-0.3
                else:
                    random_list=random.sample(range(self.n_features),2*self.s)
                    delta[random_list]=-0.5
                coef=coef_true+delta
                noise_mean = 0
                noise_var = 1
                X, y = coef_gen(coef, cov, noise_mean, noise_var, self.n_samples)
                samples_packs.append(samples_pack(X, y))
            self.samples_packs=samples_packs
            self.method='t21'
        #模型拟合
        return self.eval(estiminator, self.samples_packs,coef_true)
    
    def t31_eval(self,estiminator,h,refresh=False):
        coef_true = np.zeros(self.n_features)
        coef_true[:self.s] = 0.3
        if refresh or self.samples_packs==None or self.method!='t31':
            samples_packs=[]
            # 设定超参数：各模型回归系数、特征间的协方差矩阵、噪声的均值与方差
            # 长度为n_features, 其中前s个为非零回归系数，后n_features-s个为零
            #生成样本
            for i in range(self.n_packs):
                
                cov=None
                if i<1:
                    cov = np.eye(self.n_features)
                else:
                    cov = np.zeros((self.n_features,self.n_features))
                    for j in range(self.n_features):
                        for k in range(self.n_features):
                            if k==j:
                                cov[j][k]=1
                            elif np.abs(j-k)<2*i+1:
                                cov[j][k]=1/(i+2)
                
                delta = np.zeros(self.n_features)
                #在0~n_features之间随机选取h个位置，将其回归系数减0.3
                if i==0:
                    pass
                elif i<self.L+1:
                    random_list=random.sample(range(self.n_features),h)
                    delta[random_list]=-0.3
                else:
                    random_list=random.sample(range(self.n_features),2*self.s)
                    delta[random_list]=-0.5
                coef=coef_true+delta
                noise_mean = 0
                noise_var = 1
                X, y = coef_gen(coef, cov, noise_mean, noise_var, self.n_samples)
                samples_packs.append(samples_pack(X, y))
            self.samples_packs=samples_packs
            self.method='t31'
        #模型拟合
        return self.eval(estiminator, self.samples_packs,coef_true)