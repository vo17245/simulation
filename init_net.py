import tools

#set net param
net_size=[[4,256],[256,1024],[1024,512],[512,tools.DESITY_AXIS_LEN]]
activation_type=tools.ACTIVATION_RELU
net_param=tools.NetParam(net_size,activation_type,None)
#create net
net=tools.create_net(net_param)
#print(net.state_dict())
#save state_dict
net_param.net_state_dict=net.state_dict()
#save net param
tools.save_net_param(net_param,"net_param")
# #load net param
# net_param=tools.load_net_param("net_param")
# #deserialization net
# net=tools.create_net(net_param)
# print(net.state_dict())
