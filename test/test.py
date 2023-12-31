from ..evaluator.evaluator import *
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
from abc import abstractmethod
import numpy as np
from sklearn.linear_model import Lasso

class estiminator:
    def __init__(self,n_features):
        self.params=np.zeros(n_features)
        self.n_features=n_features
    
    @abstractmethod
    def fit(self, samples_packs,s,L):
        pass
    
    def get_params(self):
        return self.params

class our_method(estiminator):
    def __init__(self, n_features):
        super(our_method, self).__init__(n_features)
      
    #这个方法没有用到，目的是方便使用梯度下降等方法，将更新方法放到模型里面去  
    def likelihood(self, beta, delta, samples_pack):
        beta=beta+delta
        X=samples_pack.getX()
        y=samples_pack.getY()
        # 似然函数为高斯分布
        return -np.sum((y-np.dot(X,beta))**2)
    
    #更新beta
    #输入：delta:辅助模型与目标模型的回归系数差矩阵，v:辅助模型是否被选择的向量，samples_packs:样本数据
    # 优化问题：argmax_{beta} l_0(beta)+sum_{k=1}^{K}(l_k(beta,delta[k])-lambda*||delta||_2)*v[k]
    def update_beta(self, delta, v, samples_packs):#加入了l2正则-?
        # 对于线性模型，该问题有显式解
        # 该显式解为beta=(x0^T*x0+sum_{k=1}^{K}v[k]*xk^T*xk-lambda*E)^{-1}*(x0^T*y0+sum_{k=1}^{K}v[k]*xk^T*yk-sum_{k=1}^{K}v[k]*xk^T*xk*delta[k])
        # x0,y0为目标模型的样本数据
        # xk,yk为第k个辅助模型的样本数据
        
        #mat1=x0^T*x0+sum_{k=1}^{K}v[k]*xk^T*xk-lambda*E
        lamb=0.001
        mat1=np.dot(samples_packs[0].getX().T,samples_packs[0].getX())
        mat1-=lamb*np.eye(len(samples_packs[0].getX()[0]))
        for i in range(len(samples_packs)-1):
            if v[i]==1:
                mat1+=np.dot(samples_packs[i+1].getX().T,samples_packs[i+1].getX())
        #mat2=x0^T*y0+sum_{k=1}^{K}v[k]*xk^T*yk
        mat2=np.dot(samples_packs[0].getX().T,samples_packs[0].getY())
        for i in range(len(samples_packs)-1):
            if v[i]==1:
                mat2+=np.dot(samples_packs[i+1].getX().T,samples_packs[i+1].getY())
        #mat3=sum_{k=1}^{K}v[k]*xk^T*xk*delta[k]
        mat3=np.zeros(len(samples_packs[0].getX()[0]))
        for i in range(len(samples_packs)-1):
            if v[i]==1:
                mat3+=np.dot(samples_packs[i+1].getX().T,np.dot(samples_packs[i+1].getX(),delta[i]))
        beta=np.dot(np.linalg.inv(mat1),mat2-mat3)
        return beta
    
    #更新delta
    #输入：beta:当前beta，sample_pack:该模型的样本数据
    #输出：更新后的delta_k
    def update_delta_k(self,beta,sample_pack):#修改为2范数不平方    
        # 对于线性模型，这里是delta的岭回归解-?关于delta的惩罚项，ppt里面是2范数，这里先用2范数的平方了
        # 该显示解delta=(X^T*X-lambda*I)^{-1}*X^T*(y-X*beta)
        # 关于lambda的取值问题该怎么解决呢？-?这里先假定lambda=1
        lamb=0.001
        X=sample_pack.getX()
        y=sample_pack.getY()
        delta=np.dot(np.linalg.inv(np.dot(X.T,X)-lamb*np.eye(len(X[0]))),np.dot(X.T,y)-np.dot(X.T,np.dot(X,beta)))
        return delta
        
    
    # 非iid分布式迁移学习与数据源选择
    # delta[k]=beta[k]-beta[0] 为第k个辅助模型与目标模型的回归系数差
    # 上述参数均为n_features维向量
    # v[k]属于{0,1},为第k个辅助模型是否被选择
    # 优化问题：argmax_{beta,delta[k],v[k]} l_0(beta)+sum_{k=1}^{K}(l_k(beta,delta[k])-lambda*||delta||_2)*v[k]
    # s.t. ||v||_0=L
    # l_k为模型的似然函数，当前模型为线性模型，似然函数为高斯分布
    # 使用两步迭代算法求解
    # 第一步：固定v[k]，求解beta[0],delta[k]
    # 求解beta与delta时也使用两步迭代算法
    # 第二步：固定beta[0],delta[k]，求解v[k]
    # 重复以上两步直到收敛（或达到最大迭代次数）
    # -?没有设置收敛条件，目前仅仅设置了最大迭代次数
    # 输入：samples_packs:样本数据，s:目标模型稀疏度，L:选定的辅助模型个数
    def fit(self, samples_packs,s,L):
        #如果model_num==0,则直接使用目标模型的样本数据进行lasso估计
        model_num=len(samples_packs)-1
        if model_num==0:
            lasso=Lasso(alpha=0.01)
            lasso.fit(samples_packs[0].getX(),samples_packs[0].getY())
            #保留绝对值前s大个-?有意义吗？
            self.params=lasso.coef_
            return lasso.coef_,[],[],[]
        # 初始化参数
        # -?第一次迭代时使用全部模型,v=[1,1,1,1,...]
        #计算程序运行时间
        start_time=time.time()
        times=[]
        times.append(time.time()-start_time)
        
        v=np.ones(len(samples_packs)-1)
        # beta为目标模型的回归系数
        beta=np.zeros(len(samples_packs[0].getX()[0]))
        delta=np.zeros((len(samples_packs)-1,len(samples_packs[0].getX()[0])))
        # 迭代求解
        max_iter=5
        # 设置算法的退出阈值threshold
        threshold=0.0001
        beta1=np.ones(len(samples_packs[0].getX()[0]))
        beta2=np.ones(len(samples_packs[0].getX()[0]))
        for i in range(max_iter):
            time.sleep(0.5)
            # 第一步:交替求解beta与delta
            if i==0:
                times.append(time.time()-start_time)
            for j in range(max_iter):
                beta=self.update_beta(delta,v,samples_packs)
                # print(beta)
                #计算beta与beta2的差的2范数
                if np.linalg.norm(beta-beta2)<threshold:
                    break
                beta2=beta
                if i==0:
                    times.append(time.time()-start_time)
                #对K个辅助模型分别更新其delta_k
                for k in range(len(samples_packs)-1):
                    delta[k]=self.update_delta_k(beta,samples_packs[k+1])
                    # print(delta[k])
                if i==0:
                    times.append(time.time()-start_time)
            if np.linalg.norm(beta-beta1)<threshold:
                break
            beta1=beta
            # 第二步：更新v
            # 选择delta的q范数最小的L个模型，将其对应的v设为1，其余设为0。
            # 目前选择q=2
            if i==0:
                times.append(time.time()-start_time)
            v=np.zeros(len(samples_packs[1:]))
            v[np.argsort(np.linalg.norm(delta,axis=1))[:L]]=1
            print(v)
            if i==0:
                times.append(time.time()-start_time)
        # 返回beta
        # -?暂时不设置稀疏度-?设置了一下
        #保留绝对值前s大个
        # beta[np.argsort(np.abs(beta))[:-s]] = 0
        self.params=beta
        return beta,delta,v,times
    
n_features=500
n_samples=100
n_packs=21
s=16
h=6
L=2
eval=evaluator(n_features, n_packs, n_samples, s, L)
esti=our_method(n_features)
_,model_sse,model_time=eval.t11_eval(esti,h)
print(model_sse,model_time)
print(esti.get_params())