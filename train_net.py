import tools
import sys

#get args 
argc=len(sys.argv)
rounds=None
lr=0.001
if argc==2:
    rounds=int(sys.argv[1])
elif argc==3:
    rounds=int(sys.argv[1])
    lr=float(sys.argv[2])
else:
    print("invalid input")
    print("usage:\npython train_net.py rounds [lr]")
    exit()
features,labels=tools.load_data()
tools.log("map labels...")
labels_desity_map=tools.LabelsDesityMap(tools.MAX,tools.MIN,tools.STEP,labels)
tools.log("map success!")
#load net param
net_param=tools.load_net_param("net_param")
#deserialization net
net=tools.create_net(net_param)
tools.train(features,labels_desity_map.desity,net,rounds,tools.create_trainer(net,lr),40,tools.create_loss_fn(tools.LOSS_FN_MSE))
net_param.net_state_dict=net.state_dict()
tools.save_net_param(net_param,"net_param")
