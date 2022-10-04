import tools

features,labels=tools.load_data("features_test.csv","labels_test.csv")
#load net param
net_param=tools.load_net_param("net_param")
#deserialization net
net=tools.create_net(net_param)

y=net(features)

y_avg=tools.desity_average(y,tools.STEP)
labels_avg=tools.get_labels_average(labels)
print(y_avg)
print(labels_avg)
