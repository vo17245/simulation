import torch
import pandas
import numpy
import sys



#------------labels desity map param---------
MAX=1000
MIN=0
STEP=10
DESITY_AXIS_LEN=int((MAX-MIN)/STEP)
#----------------------------------------------

# -------creat_loss_fn() param-------
LOSS_FN_DESITY=0
LOSS_FN_MSE=1
LOSS_FN_L2=2
# -----------------------------------

#----------create_net() param---------
# set activation function type
ACTIVATION_SOFTMAX=0
ACTIVATION_RELU=1
#-----------------------------------

def log(msg=""):
    print (sys._getframe().f_code.co_filename,": ",sys._getframe().f_lineno,"\n",msg)

def zeros(tensor_size):
    return torch.zeros(tensor_size).detach()

def desity_average(desity,step):
    desity_size=desity.size()
    cnt=desity_size[0]
    len=desity_size[1]
    average=zeros((cnt,1))
    for i in range(cnt):
        sum=0
        cnt=0
        for j in range(len):
            a=desity[i][j]
            if a<0 :
                a=0
            b=step*j+step/2
            cnt+=a
            sum+=a*b
        average[i]=sum/cnt 
    return average



#(string,string)
def load_data(features_path="features.csv",labels_path="labels.csv"):
    d_features=pandas.read_csv(features_path,index_col=0)
    d_labels=pandas.read_csv(labels_path,index_col=0)
    a_features=numpy.array(d_features)
    a_labels=numpy.array(d_labels)
    features=torch.tensor(a_features).to(torch.float32)
    labels=torch.tensor(a_labels).to(torch.float32)
    return features,labels

#(torch.tensor,torch.tensor,int)
def get_batch(features,labels,batch_size):
    features_size=features.size()
    features_size=list(features_size)
    labels_size=labels.size() 
    labels_size=list(labels_size)
    example_cnt=features_size[0]
    features_size[0]=batch_size
    labels_size[0]=batch_size
    features_size=tuple(features_size)
    labels_size=tuple(labels_size)
    batch_features=zeros(features_size)
    batch_labels=zeros(labels_size)
    perm=torch.randperm(example_cnt)
    for i in range(batch_size):
        batch_features[i]=features[perm[i]]
        batch_labels[i]=labels[perm[i]]
    return batch_features,batch_labels

#(torch.nn.module,double)
def create_trainer(net,lr):
    trainer = torch.optim.RMSprop(net.parameters(), lr,weight_decay=1e-3)
    return trainer

def create_loss_fn(type=LOSS_FN_MSE):
    fn=None
    if type==LOSS_FN_DESITY:
        pass
    elif type==LOSS_FN_MSE:
        fn=torch.nn.MSELoss(reduction='sum')
    return fn

#(torch.tensor,torch.tensor,torch.nn.module,torch.optim)
def train(features,labels,net,rounds,trainer,batch_size,loss_fn):
    for i in range(rounds):
        batch_features,batch_labels=get_batch(features,labels,batch_size)
        y_pred=net(batch_features)
        loss=loss_fn(y_pred,batch_labels)

        print(i,"/",rounds,loss)
        trainer.zero_grad()
        loss.backward()
        trainer.step()


class LabelsDesityMap:
    max=None
    min=None
    step=None
    #len of desity_axis
    len=None
    desity_axis=None
    desity=None
    __average=None
    labels_cnt=None
    def __init__(self,max,min,step,labels):
        self.max=max
        self.min=min
        self.step=step
        self.len=(max-min)/step
        self.len=int(self.len)
        self.desity_axis=zeros((self.len,1))
        for i in range(self.len):
            self.desity_axis[i]=step*i
        labels_size=labels.size()
        self.labels_cnt=labels_size[0]
        label_len=labels_size[1]
        self.desity=zeros((self.labels_cnt,self.len))
        
        for i in range(self.labels_cnt):
            for j in range(label_len):
                val=labels[i][j]
                flag=0
                for k in range(self.len):
                    if val>self.desity_axis[k]:
                        if val<=self.desity_axis[k]+step:
                            self.desity[i][k]+=1
                            flag=1
                            break
                

    def average(self):
        if self.__average != None:
            return self.__average
        for i in range(self.labels_cnt):
            sum=0
            cnt=0
            w=0
            for j in range(self.len):
               a=self.desity[i][j]
               if a<0 :
                  a=0
               b=self.desity_axis[j]+self.step/2
               cnt+=a
               sum+=a*b
            self.__average[i]=sum/cnt 
        return self.__average

def get_labels_average(labels):
    labels_size=labels.size()
    labels_cnt=labels_size[0]
    label_len=labels_size[1]
    average=zeros((labels_cnt,1))
    for i in range(labels_cnt):
        sum=0
        for  j in range(label_len):
            sum+=labels[i][j]
        average[i]=sum/label_len
    return average

def print_tensor(t):
    print(t.detach().numpy().tolist())

def create_net(param):
    activation_type=param.activation_type
    net_size=param.net_size
    activation_module=None
    if activation_type==ACTIVATION_RELU:
        activation_module=torch.nn.ReLU()
    elif activation_type==ACTIVATION_SOFTMAX:
        activation_module=torch.nn.Softmax()
    net=torch.nn.Sequential()
    layer_cnt=len(net_size)
    for i in range(layer_cnt-1):
        hidden=torch.nn.Sequential(torch.nn.Linear(net_size[i][0],net_size[i][1]),activation_module)
        net=torch.nn.Sequential(net,hidden)
    hidden=torch.nn.Sequential(torch.nn.Linear(net_size[layer_cnt-1][0],net_size[layer_cnt-1][1]))
    net=torch.nn.Sequential(net,hidden)
    if param.net_state_dict!=None:
        net.load_state_dict(param.net_state_dict)
    return net

def save_net_param(param,path):
    torch.save(param,path)

def load_net_param(path):
    param=torch.load(path)
    return param

#class for saving net (by serialization)
class NetParam:
    net_state_dict=None
    net_size=None
    activation_type=None
    def __init__(self,net_size,activation_type,net_state_dict):
        self.net_stae_dict=net_state_dict
        self.net_size=net_size
        self.activation_type=activation_type
        pass

