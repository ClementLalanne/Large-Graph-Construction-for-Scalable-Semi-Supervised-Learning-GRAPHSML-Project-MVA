import random
import numpy as np
import myssl
import time

def kthcenter(k,n):
    return np.cos(k*2*np.pi/n), np.sin(k*2*np.pi/n)


def kblobs(var=0.2, k=2, n=100):
    data = []
    labels = []
    means = [kthcenter(i,k) for i in range(k)]
    for i in range(n):
        indx = int(random.random()*k)
        data.append([random.gauss(means[indx][0],var), random.gauss(means[indx][1],var)])
        labels.append(indx+1)
    return np.array(data), np.array(labels)

def test(var, k, n, m, s, p):
    data, labels = kblobs(var, k, n)
    hidlabs = myssl.hide_labels(labels,p)
    agr = myssl.SSLSolver()
    predlabels,_,_ = agr.fit(data, hidlabs, m, s)
    print(labels[:20])
    print(predlabels[:20])
    score=0
    for i in range(n):
        if labels[i]==predlabels[i]:
            score+=1
    print("Score is {} out of {}".format(score, n))

start = time.time()
test(1,5,1000,10,6,50)
print(time.time()-start)
