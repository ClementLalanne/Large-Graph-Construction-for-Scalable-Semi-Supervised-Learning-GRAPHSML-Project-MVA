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
import tensorflow as tf

np.random.seed(10)

print("Importing CIFAR")

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
        
cifar = cifar10loader('data/cifar-10-batches-py')

train_data = cifar.data
train_labels = cifar.labels + 1

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
    l1 = tf.nn.tanh(fc_layer(x, 32*32*3, 100))
    l2 = tf.nn.tanh(fc_layer(l1, 100, 100))
    l3 = fc_layer(l2, 100, REDUCED_SIZE)
    l4 = tf.nn.tanh(fc_layer(l3, REDUCED_SIZE, 100))
    l5 = tf.nn.tanh(fc_layer(l4, 100, 100))
    out = tf.nn.relu(fc_layer(l5, 100, 32*32*3))
    loss = tf.reduce_mean(tf.squared_difference(x, out))
    return loss, out, l3

BATCH_SIZE = 32

print("Training Autoencoder")


with tf.Session() as sess:
    x = tf.placeholder(tf.float32, shape=[None, 32*32*3])

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
    data, labels = train_data_reduced[:1000, :], train_labels[:1000]
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









