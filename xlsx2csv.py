import numpy
import pandas
import torch

perm=None
r=0.9
def rand_divide(data,r):
    data_size=data.shape
    data_cnt=data_size[0]
    len=data_size[1]
    cnt1=int(r*data_cnt)
    cnt2=data_cnt-cnt1
    p1=numpy.zeros((cnt1,len))
    p2=numpy.zeros((cnt2,len))
    for i in range(cnt1):
        p1[i]=data[perm[i]]
    for i in range(cnt2):
        p2[i]=data[perm[cnt1+i]]
    return p1,p2

#load features.xlsx
d_features=pandas.read_excel("features.xlsx",header=None)
#change columns
a_features=numpy.array(d_features)
example_cnt=a_features.shape[0]
perm=torch.randperm(example_cnt)
p1,p2=rand_divide(a_features,r)
a_features=p1
a_features_test=p2
d_features=pandas.DataFrame(a_features,columns=["浓度(%)","电压(kV)","接近距离(cm)","推进速度(mL/h)"])
d_features_test=pandas.DataFrame(a_features_test,columns=["浓度(%)","电压(kV)","接近距离(cm)","推进速度(mL/h)"])
#save features.csv
d_features.to_csv("features.csv")
d_features_test.to_csv("features_test.csv")

#load labels.xlsx
d_labels=pandas.read_excel("labels.xlsx",header=None)
#transfer to numpy array
a_labels=numpy.array(d_labels)
#transpose 
a_labels=a_labels.transpose()
#caculate average for every row
shape=a_labels.shape
averages=numpy.zeros(shape[0])
for i in range(shape[0]):
    sum=0
    cnt=0
    for j in range(shape[1]):
        if numpy.isnan(a_labels[i][j]):
            pass
        else:
            sum+=a_labels[i][j]
            cnt+=1
    averages[i]=sum/cnt
#replace nan element with average
for row in range(shape[0]):
    for col in range(shape[1]):
        if(numpy.isnan(a_labels[row][col])):
            a_labels[row][col]=averages[row]

#save labels.csv
p1,p2=rand_divide(a_labels,r)
a_labels=p1
a_labels_test=p2
pandas.DataFrame(a_labels).to_csv("labels.csv")
pandas.DataFrame(a_labels_test).to_csv("labels_test.csv")


