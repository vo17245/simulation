import tools

features,labels=tools.load_data("features_test.csv","labels_test.csv")
desity=tools.LabelsDesityMap(tools.MAX,tools.MIN,tools.STEP,labels)
labels=desity.desity
#load net param
net_param=tools.load_net_param("net_param")
#deserialization net
net=tools.create_net(net_param)
y=net(features)
loss_fn=tools.create_loss_fn()
loss=loss_fn(labels,y)
print(loss)