# imports
import time
import numpy as np
import pickle
import myssl
import pandas as pd

import matplotlib.pyplot as plt


path="data/" 

Dtrain=pd.read_csv(path+"mnist_train.csv",header=None)

Ltrain=Dtrain.iloc[:,0]
Dtrain.drop(Dtrain.columns[[0]], axis=1,inplace=True)

train_data = np.array(Dtrain.iloc[:,:], dtype=float)/255
train_labels = np.array(Ltrain.iloc[:], dtype=int)

def test(m, s, p):
    data, labels = train_data[:10000, :], train_labels[:10000]+1
    hidlabs = myssl.hide_labels(labels,p)
    agr = myssl.SSLSolver()
    predlabels,_,_ = agr.fit(data, hidlabs, m, s)
    print(labels[:20])
    print(predlabels[:20])
    score=0
    for i in range(len(labels)):
        if labels[i]==predlabels[i]:
            score+=1
    print("Score is {} out of {}".format(score, len(labels)))

start = time.time()
test(100,5,50)
print(time.time()-start)
