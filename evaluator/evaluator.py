# 该类是一个分布式联合估计下的评估器，用于评估联合估计的性能
# 该类的输入是一个联合估计器，输出是一个评估结果
# 联合估计器的输入是来自于K+1个不同线性模型的观测数据，其中包括一个目标模型与K个辅助模型。输出是估计目标模型的参数
# 这K+1个模型使用相同的特征，但其样本的分布、噪声的分布、模型的参数都可能不同，特征之间的相关性也可能不同

import time
import numpy as np

#加入了多线程的评估器，
class evaluator:
    def __init__(self,repeat_times=1,model_num=1) -> None:
        self.repeat_times=repeat_times
        self.task_queue=[]
        for i in range(model_num):
            self.task_queue.append([])
        self.result_list=[]
        self.count=0
        self.model_num=model_num
        
    def __task_len__(self):
        length=0
        for i in range(self.model_num):
            length+=len(self.task_queue[i])
        return length
    
    # 对给定的算法、样本数据、真实参数进行一次评估
    def eval(self, estiminator, sample_packs,coef_true,s,L,index=None,model_index=None):
        #计时
        start=time.time()
        estiminator.fit(sample_packs,s,L)
        end=time.time()
        SSE=np.sum((coef_true-estiminator.get_params())**2)
        # print(index)
        #将结果输出到result_list中
        return end-start,SSE,index,model_index
    
    def append(self,method_list,samples_packs,coef_true,s,L,repeat=False):#-?重复的逻辑写错了，以后有缘修改一下
        #如果method_list 的类型为dic，则将其转换为list
        if type(method_list)==dict:
            method_list=list(method_list.values())
        for i in range(self.model_num):
            if repeat:
                for j in range(self.repeat_times):
                    self.task_queue[i].append((method_list[i],samples_packs,coef_true,s,L,len(self.task_queue[i]),i))
            else:
                self.task_queue[i].append((method_list[i],samples_packs,coef_true,s,L,len(self.task_queue[i]),i))

    def run(self,max_workers=1,multi_thread=True):
        #多线程，线程数为cpu核心数
        if multi_thread:
            import multiprocessing
            self.result_list=[]
            for i in range(self.model_num):
                self.result_list.append([None]*len(self.task_queue[i]))
                for j in range(1+int(len(self.task_queue[i])/max_workers)):
                    pool=multiprocessing.Pool(max_workers)
                    for task in self.task_queue[i][j*max_workers:(j+1)*max_workers]:
                        pool.apply_async(self.eval,task,callback=self.callback)
                    pool.close()
                    pool.join()
        else:
            self.result_list=[]
            for i in range(self.model_num):
                self.result_list.append([None]*len(self.task_queue[i]))
                for j in range(len(self.task_queue[i])):
                    self.callback(self.eval(*self.task_queue[i][j]))
    

    def callback(self,result):
        times,SSE,index,model_index=result
        self.result_list[model_index][index]=SSE
        self.count+=1
        # print(str(self.count)+'/'+str(self.__task_len__()))
        
    def get_result(self):
        return 0