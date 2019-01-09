# imports
import os
import sys

cwd = os.getcwd()
sys.path.insert(0,cwd+'/..')
import time
import numpy as np
import pickle
import myssl
import matplotlib.pyplot as plt

np.random.seed(10)

#%%
def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1).reshape(((len(batch['data']))), 3*32*32)/255
    labels = batch['labels']
        
    return features, labels
#%%
def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x 
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x
#%%
class cifar10loader():
    def __init__(self, path):
        
        self.data = np.zeros((0, 3*32*32))
        self.labels = np.zeros(0, dtype=int)
        self.size = 0
        
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            features, labels = load_cfar10_batch(path, batch_i)
            features = normalize(features)
            #labels = one_hot_encode(labels)
            self.data = np.concatenate((self.data, features), axis=0)
            self.labels = np.concatenate((self.labels, np.array(labels, dtype=int)), axis=0)
        self.size = self.labels.shape[0]
        
cifar = cifar10loader(cwd + '/../' + 'data/cifar-10-batches-py')

def test(m, s, p):
    data, labels = cifar.data[:1000, :], cifar.labels[:1000]+1
    hidlabs = myssl.hide_labels(labels,p)
    agr = myssl.SSLSolver()
    predlabels,_,_ = agr.fit(data, hidlabs, m, s, tuning_param=10.)
    print(labels[:20])
    print(predlabels[:20])
    score=0
    for i in range(len(labels)):
        if labels[i]==predlabels[i]:
            score+=1
    print("Score is {} out of {}".format(score, len(labels)))

start = time.time()
test(100,50,50)
print(time.time()-start)









