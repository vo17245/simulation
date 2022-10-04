import numpy
import pandas

d_features=pandas.read_csv("features.csv",index_col=0)
a_features=numpy.array(d_features)

d_labels=pandas.read_csv("labels.csv",index_col=0)
a_labels=numpy.array(d_labels)
print(a_labels)