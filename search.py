import tools
import torch
import numpy

#--------------global-----------------------

#load net param
net_param=tools.load_net_param("net_param")
#deserialization net
net=tools.create_net(net_param)

#set search param

# search space for class Code
C_SIZE=[0,20,0.01]#min max step ----> [min,max)
U_SIZE=[0,20,0.01]
D_SIZE=[0,20,0.01]
V_SIZE=[0,2,0.01]

C_LEN=(C_SIZE[1]-C_SIZE[0])/C_SIZE[2]
C_LEN=(C_SIZE[1]-C_SIZE[0])/C_SIZE[2]
C_LEN=(C_SIZE[1]-C_SIZE[0])/C_SIZE[2]


rounds=200

target=300
#-----------------------------------------

#------------class---------------------

class Code:
    c=None
    u=None
    d=None
    v=None
    data=None
    def __init__(self,c,u,d,v):
        pass


#------------------------------------

#---------------function------------

def predict(x):
    x_t=torch.tensor(x)
    y=net(x_t)
    y_avg=tools.desity_average(y,tools.STEP)
    return y_avg

def loss_fn(x):
    avg=predict(x)
    return abs(target-avg)

def encode(x):
    pass
#-----------------------------












    
