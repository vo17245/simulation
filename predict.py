import tools
import torch
import sys

argv=sys.argv
argc=len(argv)
if argc!=5:
    print("ERROR: invalid input!")
    exit()
a=float(argv[1])
b=float(argv[2])
c=float(argv[3])
d=float(argv[4])


x=[[a,b,c,d]]
x=torch.tensor(x)
#load net param
net_param=tools.load_net_param("net_param")
#deserialization net
net=tools.create_net(net_param)

y=net(x)

y_avg=tools.desity_average(y,tools.STEP)
print(y_avg)

