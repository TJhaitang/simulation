from methods.estiminator import estiminator
import numpy as np
from sklearn.linear_model import Lasso
import time

# 该估计器仅使用目标模型的样本数据进行lasso估计
class T_lasso(estiminator):
    def __init__(self, n_features=0,s=0,L=0,instance=None):
        if instance is not None:
            super(T_lasso, self).__init__(instance.n_features,instance.s,instance.L)
        else:
            assert n_features>0 and s>0 and L>0
            super(T_lasso, self).__init__(n_features,s,L)
        
    def fit(self,samples_packs,s=0,L=0):
        from sklearn.linear_model import Lasso
        lambda1=0.01
        X=samples_packs[0].getX()
        y=samples_packs[0].getY()
        lasso=Lasso(alpha=lambda1)
        lasso.fit(X,y)
        #保留绝对值前s大个-?有意义吗？
        # lasso.coef_[np.argsort(np.abs(lasso.coef_))[:-self.s]] = 0
        self.params=lasso.coef_
        return lasso.coef_