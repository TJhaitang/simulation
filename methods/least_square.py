from methods.estiminator import estiminator
import numpy as np
import time

# 该估计器仅使用目标模型的样本数据进行lasso估计
class Least_square(estiminator):
    def __init__(self, n_features=0,s=0,L=0,instance=None):
        if instance is not None:
            super(Least_square, self).__init__(instance.n_features,instance.s,instance.L)
        else:
            assert n_features>0 and s>0 and L>0
            super(Least_square, self).__init__(n_features,s,L)
        
    def fit(self,samples_packs,s=0,L=0):
        X=samples_packs[0].getX()
        y=samples_packs[0].getY()
        #直接最小二乘
        coef=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        #保留绝对值前s大个-?有意义吗？
        # coef.coef_[np.argsort(np.abs(coef.coef_))[:-self.s]] = 0
        self.params=coef
        return coef