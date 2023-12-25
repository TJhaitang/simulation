from matplotlib import pyplot as plt

def plot_graphs(model_dic,result_list,h_list,L_list):
    fig=plt.figure(figsize=(20,5))
    for i in range(len(h_list)):
        ax=fig.add_subplot(1,len(h_list),i+1)
        ax.set_title('h='+str(h_list[i]))
        for j in range(len(model_dic)):
            ax.plot(L_list,result_list[i,j,:],label=list(model_dic.keys())[j])
        ax.legend()
        ax.set_xlabel('L')
        ax.set_ylabel('SSE')
    plt.show()