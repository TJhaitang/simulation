from abc import abstractmethod
import time
import numpy as np
from tqdm import tqdm
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
    
    
# 该估计器仅使用目标模型的样本数据进行lasso估计
class t_lasso(estiminator):
    def __init__(self, n_features):
        super(t_lasso,self).__init__(n_features)
        
    def fit(self,samples_packs,s,L):
        from sklearn.linear_model import Lasso
        lambda1=0.01
        X=samples_packs[0].getX()
        y=samples_packs[0].getY()
        lasso=Lasso(alpha=lambda1)
        lasso.fit(X,y)
        #保留绝对值前s大个-?有意义吗？
        lasso.coef_[np.argsort(np.abs(lasso.coef_))[:-s]] = 0
        self.params=lasso.coef_
        return lasso.coef_
    
# 这是一个联合估计器，使用的算法来自Transfer learning for high-dimensional linear regression: Prediction, estimation and minimax optimality
class trans_lasso(estiminator):
    def __init__(self, n_features):
        super(trans_lasso, self).__init__(n_features)
    
    #关于lambda的选择问题，原始论文中是这样进行的：
    #R:
    #cv.init<-cv.glmnet(X[ind.kA,], y.A, nfolds=8, lambda=seq(1,0.1,length.out=10)*sqrt(2*log(p)/length(ind.kA)))
    #lam.const <- cv.init$lambda.min/sqrt(2*log(p)/length(ind.kA))
    #lambda=lam.const*sqrt(2*log(p)/length(ind.kA)))$beta
    def fit(self, samples_packs,s,L):
        #Step1-划分训练集与测试集
        model_num=len(samples_packs)-1
        if model_num==0:
            #直接lasso
            lasso=Lasso(alpha=0.01)
            lasso.fit(samples_packs[0].getX(),samples_packs[0].getY())
            #保留绝对值前s大个-?有意义吗？
            self.params=lasso.coef_
            return lasso.coef_,[],[],[]
        X0=samples_packs[0].getX()
        y0=samples_packs[0].getY()
        #按照7:3划分
        trainX=X0[:int(len(X0)*0.7)]
        trainy=y0[:int(len(y0)*0.7)]
        testX=X0[int(len(X0)*0.7):]
        testy=y0[int(len(y0)*0.7):]
        #Step2-生成GL下标集合
        #deltak=1/n_k*sum_{i=1}^{n_k}(x_i^T*y_i)-1/|X_0|*sum_{i=1}^{|X_0|}(x_i^T*y_i)
        R=[]
        t_star=20
        for i in range(len(samples_packs)-1):
            delta=np.dot(samples_packs[i+1].getX().T,samples_packs[i+1].getY())/len(samples_packs[i+1])-np.dot(trainX.T,trainy)/len(trainX)
            #将delta的绝对值从大到小排序，取前t_star个，其余置零作为新的delta
            delta[np.argsort(np.abs(delta))[:-t_star]]=0
            #delta的2范数的平方作为R
            R.append(np.linalg.norm(delta)**2)
        #从R中找出最小的L个元素对应的下标，作为GL的下标集合
        GL=np.argsort(R)[:L]
        #Step3-在GL中使用前1个、前2个、...、前L个模型分别进行Oracle Trans Lasso进行估计，获得beta1、beta2、...、betaL
        #Oracle Trans Lasso: 进行两步lasso估计
        #第一步：在所有的辅助模型中使用lasso进行估计，获得回归系数w
        #第二步：在目标模型中使用lasso进行估计，惩罚项为beta-w
        #这里使用sklearn中的lasso进行估计
        beta=[]
        for i in range(L):
            G=GL[:i+1]
            #第一步，在下标为G的辅助模型上同时使用lasso
            lasso=Lasso(alpha=0.01)
            infoX=[]
            infoy=[]
            for j in range(len(G)):
                infoX.append(samples_packs[G[j]+1].getX())
                # print(samples_packs[G[j]+1].getX().shape)
                infoy.append(samples_packs[G[j]+1].getY())
            infoX=np.concatenate(infoX)
            infoy=np.concatenate(infoy)
            # print(infoX.shape,infoy.shape)
            lasso.fit(infoX,infoy)
            w=lasso.coef_
            assert len(w)==self.n_features
            #第二步，在目标模型上使用lasso估计beta，但惩罚项为beta-w
            lasso=Lasso(alpha=0.01)
            trainyL=trainy-np.dot(trainX,w)
            lasso.fit(trainX,trainyL)
            beta.append(lasso.coef_+w)
        #Step4-使用beta1、beta2、...、betaL进行估计，获得beta
        #选择在测试集上表现最好的beta作为最终的beta
        from sklearn.metrics import mean_squared_error
        min_error=mean_squared_error(testy,np.dot(testX,beta[0]))
        min_index=0
        for i in range(L):
            error=mean_squared_error(testy,np.dot(testX,beta[i]))
            if error<min_error:
                min_error=error
                min_index=i
        #Step5-取绝对值前s大
        # beta[min_index][np.argsort(np.abs(beta[min_index]))[:-s]] = 0  
        self.params=beta[min_index]
        assert len(self.params)==self.n_features
        return beta[min_index],GL
            
        
# 
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
        max_iter=10
        # 设置算法的退出阈值threshold
        threshold=0.001
        beta1=np.ones(len(samples_packs[0].getX()[0]))
        beta2=np.ones(len(samples_packs[0].getX()[0]))
        for i in range(max_iter):
            # 第一步:交替求解beta与delta
            if i==0:
                times.append(time.time()-start_time)
            for j in range(max_iter):
                beta=self.update_beta(delta,v,samples_packs)
                #计算beta与beta2的差的2范数
                if np.linalg.norm(beta-beta2)<threshold:
                    break
                beta2=beta
                if i==0:
                    times.append(time.time()-start_time)
                #对K个辅助模型分别更新其delta_k
                for k in range(len(samples_packs)-1):
                    delta[k]=self.update_delta_k(beta,samples_packs[k+1])
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
            if i==0:
                times.append(time.time()-start_time)
        # 返回beta
        # -?暂时不设置稀疏度-?设置了一下
        #保留绝对值前s大个
        # beta[np.argsort(np.abs(beta))[:-s]] = 0
        self.params=beta
        return beta,delta,v,times