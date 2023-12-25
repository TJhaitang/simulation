from methods.our_method import Our_method
from evaluator.evaluator import *
from evaluator.sample_generator import *
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import numpy as np

n_features=16
n_samples=100
n_packs=101
s=16
#evaluator.t11_eval(alg)
# A_alg=our_method(n_features)
# Trans_lasso=trans_lasso(n_features)
# lasso=t_lasso(n_features)
# models=[our_method(n_features),trans_lasso(n_features),t_lasso(n_features)]
models=[Our_method(n_features,s,1)]
SSE=np.zeros((3,3,13))
times=np.zeros((3,3,13))
#L=2,4,8,12,16,20 -?没有做L=0的代码
#多次实验取平均
for i in tqdm(range(1)):
    for h in [2,6,10]:
        for L in [2,4,8,12,16,20,24,28,32,36,40,44,48]:
            eval=evaluator(repeat_times=5)
            sample_packs,coef_true=t11_eval(n_features,s,n_packs,n_samples,h,L)
            for model in models:
                model_time,model_sse=eval.eval(model,sample_packs,coef_true,s,L)
                SSE[int(h/5),models.index(model),int(L/4)]+=model_sse
                times[int(h/5),models.index(model),int(L/4)]+=model_time
    # break
SSE=SSE/(i+1)
print(SSE)