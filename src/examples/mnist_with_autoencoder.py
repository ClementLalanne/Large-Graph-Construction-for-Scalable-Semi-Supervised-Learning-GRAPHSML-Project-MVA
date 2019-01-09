# imports
import os
import sys

cwd = os.getcwd()
sys.path.insert(0,cwd+'/..')
import time
import numpy as np
import pickle
import myssl
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt


path="data/" 

Dtrain=pd.read_csv(path+"mnist_train.csv",header=None)

Ltrain=Dtrain.iloc[:,0]
Dtrain.drop(Dtrain.columns[[0]], axis=1,inplace=True)

train_data = np.array(Dtrain.iloc[:,:], dtype=float)/255
train_labels = np.array(Ltrain.iloc[:], dtype=int)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def fc_layer(prev, input_size, output_size):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf.matmul(prev, W) + b

REDUCED_SIZE = 50

def autoencoder(x):
    l1 = tf.nn.tanh(fc_layer(x, 28*28, 100))
    l2 = tf.nn.tanh(fc_layer(l1, 100, 100))
    l3 = fc_layer(l2, 100, REDUCED_SIZE)
    l4 = tf.nn.tanh(fc_layer(l3, REDUCED_SIZE, 100))
    l5 = tf.nn.tanh(fc_layer(l4, 100, 100))
    out = tf.nn.relu(fc_layer(l5, 100, 28*28))
    loss = tf.reduce_mean(tf.squared_difference(x, out))
    return loss, out, l3

BATCH_SIZE = 32

print("Training Autoencoder")


with tf.Session() as sess:
    x = tf.placeholder(tf.float32, shape=[None, 28*28])

    loss, output, latent = autoencoder(x)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # run the training loop
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        t_size = train_data.shape[0]
        batch = train_data[np.random.permutation(np.arange(t_size))[:BATCH_SIZE], :]
        feed = {x : batch}
        if i % 500 == 0:
            print("Step: %d. " % (i))
        train_step.run(feed_dict=feed, session=sess)





    print("Building reduced set")

    train_data_reduced = np.zeros((train_data.shape[0], REDUCED_SIZE))
    Mlatent = sess.run([latent], feed_dict = {x: train_data})
    train_data_reduced = Mlatent[0]

print("Computing SSL")


def test(m, s, p):
    data, labels = train_data_reduced[:1000, :], train_labels[:1000]+1
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
