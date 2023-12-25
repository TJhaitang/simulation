from abc import abstractmethod
import time
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Lasso

class estiminator:
    def __init__(self,n_features,s=0,L=0):
        self.params=np.zeros(n_features)
        self.n_features=n_features
        self.s=s
        self.L=L
    
    @abstractmethod
    def fit(self, samples_packs,s=0,L=0):
        pass
    
    def get_params(self):
        return self.params