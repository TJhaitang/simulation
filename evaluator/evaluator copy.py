# 该类是一个分布式联合估计下的评估器，用于评估联合估计的性能
# 该类的输入是一个联合估计器，输出是一个评估结果
# 联合估计器的输入是来自于K+1个不同线性模型的观测数据，其中包括一个目标模型与K个辅助模型。输出是估计目标模型的参数
# 这K+1个模型使用相同的特征，但其样本的分布、噪声的分布、模型的参数都可能不同，特征之间的相关性也可能不同

import time
import numpy as np

#加入了多线程的评估器，
class evaluator:
    def __init__(self,repeat_times=1) -> None:
        self.repeat_times=repeat_times
        self.task_queue=[]
        self.result_list=[]
        self.count=0
        
    def __task_len__(self):
        return len(self.task_queue)
    
    # 对给定的算法、样本数据、真实参数进行一次评估
    def eval(self, estiminator, sample_packs,coef_true,s,L,index=None):
        #计时
        start=time.time()
        estiminator.fit(sample_packs,s,L)
        end=time.time()
        SSE=np.sum((coef_true-estiminator.get_params())**2)
        # print(index)
        #将结果输出到result_list中
        return end-start,SSE,index
    
    def append(self,method,samples_packs,coef_true,s,L,repeat=False):#-?重复的逻辑写错了，以后有缘修改一下
        if repeat:
            for i in range(self.repeat_times):
                self.task_queue.append((method,samples_packs,coef_true,s,L,len(self.task_queue)))
        else:
            self.task_queue.append((method,samples_packs,coef_true,s,L,len(self.task_queue)))

    def run(self,max_workers=1):
        #多线程，线程数为cpu核心数
        import multiprocessing
        pool=multiprocessing.Pool(max_workers)
        self.result_list=[None]*len(self.task_queue)
        for task in self.task_queue:
            pool.apply_async(self.eval,task,callback=self.callback)
        pool.close()
        pool.join()
    
    def callback(self,result):
        times,SSE,index=result
        self.result_list[index]=SSE
        self.count+=1
        print(str(self.count)+'/'+str(len(self.task_queue)))
        
    def get_result(self):
        return 0
    
    #  def queue_eval(self,task_queue):
    #     result_list=[]
    #     for task in task_queue:
    #         result_list.append(self.eval(*task))
    #     return result_list
    
    # def append(self,method,samples_packs,coef_true,s,L,repeat=False):#-?重复的逻辑写错了，以后有缘修改一下
    #     if repeat:
    #         for i in range(self.repeat_times):
    #             self.task_queue.append((method,samples_packs,coef_true,s,L,len(self.task_queue)))
    #     else:
    #         self.task_queue.append((method,samples_packs,coef_true,s,L,len(self.task_queue)))

    # def run(self,max_workers=1):
    #     #多线程，线程数为cpu核心数
    #     import multiprocessing
    #     pool=multiprocessing.Pool(max_workers)
    #     #将self.taskqueue分成max_workers份，每份分给一个线程
    #     self.result_list=[None]*len(self.task_queue)
    #     for i in range(max_workers):
    #         task_queue=self.task_queue[round(i/max_workers*len(self.task_queue)):round((i+1)/max_workers*len(self.task_queue))]
    #         pool.apply_async(self.queue_eval,task_queue,callback=self.queue_callback)
    #     pool.close()
    #     pool.join()
    #     # import multiprocessing
    #     # cores = multiprocessing.cpu_count()
    #     # pool = multiprocessing.Pool(processes=cores)
    #     # self.result_list=pool.map(self.eval,self.task_queue)
    #     # pool.close()
    #     # pool.join()

    # def queue_callback(self,result_list):
    #     for result in result_list:
    #         self.callback(result)


