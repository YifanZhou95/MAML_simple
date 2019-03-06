# -*- coding: utf-8 -*-
"""
Model-Agnostic Meta-Learning on mnist

https://arxiv.org/pdf/1703.03400.pdf

Created on Jan 2019

@author: zyf
"""
import numpy as np
import random
import time
from mnist import MNIST
import matplotlib.pyplot as plt

from DataLoader import DataLoader
from utils import *

mndata = MNIST('../data')
images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()


### Split data ###

n_category = 10
category_train = [0,1,2,3,4,5,6,7]
category_test = [8,9]

d = len(images_train[0])
images_train = images_train[:20000]
labels_train = labels_train[:20000]
images_test = images_test[-20000:]
labels_test = labels_test[-20000:]

images_train_categories = [[] for _ in range(n_category)]
images_test_categories = [[] for _ in range(n_category)]

for idx in range(len(images_train)):
    category = labels_train[idx]
    if category in category_train:
        images_train_categories[category].append([1]+images_train[idx])

for idx in range(len(labels_test)):
    category = labels_test[idx]
    if category in category_test:
        images_test_categories[category].append([1]+images_test[idx])
    

# initialize dataloader
trainLoaders = []
testLoaders = []

# k_shots = 2
# batch_size = 2*k_shots     # mini batch size for training

for c in range(n_category):
    if c in category_train:
        trainLoaders.append(DataLoader(np.array([(i+[c]) for i in images_train_categories[c]])))
        trainLoaders[-1].reset(0)
    if c in category_test:
        testLoaders.append(DataLoader(np.array([(i+[c]) for i in images_test_categories[c]])))
        testLoaders[-1].reset(0)



# A single episode for meta-learning (either for train or fine-tune)

def oneEpisode(is_meta=True, is_train=True):
    
    global w_meta
    w_copy = w_meta + 0  # make copy
    w_base = np.zeros((d+1,1))  # baseline parameter initialize
    gw_meta_avg, gw_base_avg = 0, 0
    
    n_task = num_task if is_train else 1
    max_iter = 1 if is_train else n_step_tuning
    
    # determine parameter
    dataLoaders = trainLoaders if is_train else testLoaders
    ctgy_range = len(category_train) if is_train else len(category_test)
    sample_size = batch_size if is_train else 0

    task_pool = []
    for i_task in range(n_task):
        task_pool.append(getSamples(dataLoaders, ctgy_range, sample_size))
        
    for iter_i in range(max_iter):
        
        for task_data in task_pool:

            inputs_alpha, targets_alpha, inputs_bravo, targets_bravo = task_data
            
            gw_meta, accu_train_meta, accu_query_meta = getGradient(inputs_alpha, targets_alpha, inputs_bravo, targets_bravo, w_meta, k_shots, learn_rate_alpha, is_train)
            gw_meta_avg += gw_meta / n_task
            
            accu_train_base, accu_query_base = 0, 0

            if is_train: continue
                
            gw_base, accu_train_base, accu_query_base = getGradient(inputs_alpha, targets_alpha, inputs_bravo, targets_bravo, w_base, k_shots, learn_rate_alpha, is_train)
            gw_base_avg += gw_base / n_task
            
            # display inside epoch
            if (iter_i + 1) % disp_round == 0:
                if (iter_i + 1) == disp_round: print("\n")
                print ("iteration:", iter_i + 1, end='\t')
                # print (loss_train, reg_loss, loss_valid, loss_test)
                print (round(accu_train_meta/(0+1), 3), round(accu_query_meta/(0+1), 3), round(accu_train_base/(0+1), 3), round(accu_query_base/(0+1), 3))
        
        learning_rate = learn_rate_beta if is_train else learn_rate_alpha
        w_meta += learning_rate * gw_meta_avg
        w_base += amplifier * learning_rate * gw_base_avg
    
    if not is_train:
        w_meta = w_copy
        
    return accu_query_meta, accu_query_base


def testProcedure():

    print("intermediate testing...")
    print(time.asctime( time.localtime(time.time()) ))
    print("\n")

    max_sample = 20
    accu_meta_avg, accu_base_avg = 0, 0

    for i in range(max_sample):
        accu_meta, accu_base = oneEpisode(is_meta=False, is_train=False)
        accu_meta_avg += accu_meta / max_sample
        accu_base_avg += accu_base / max_sample
        if (i+1)%10==0:
            print('Sample (test):', i+1, '\tAccuracy:', round(accu_meta, 3), round(accu_base, 3))
            print(time.asctime( time.localtime(time.time()) ))
            print("\n")

    print("[test]: Average accuracy:", accu_meta_avg, accu_base_avg)
    print("\n")
    return accu_meta_avg, accu_base_avg


k_shots = 1
num_task = 2
batch_size = 2*k_shots     # mini batch size for training
max_epsd = 10000
learn_rate_alpha = 5e-9    # meta update step size alpha
learn_rate_beta = 5e-9    # meta update step size beta
amplifier = 10    # indicate larger step size for baseline finetuning, should be 1 in fact ?

n_step_tuning = 16      # finetuning steps
disp_round = 4

w_meta = np.zeros((d+1,1))
loss_record = [[],[],[]]
accu_record = [[],[],[]]


print("Begin training...")
print(time.asctime( time.localtime(time.time()) ))
print("\n")

for i in range(max_epsd):
    accu_meta, _ = oneEpisode(is_meta=True, is_train=True)
    if (i+1)%100==0:
        print('Epoch (train):', i+1, '\tAccuracy:', round(accu_meta, 3))
        print(time.asctime( time.localtime(time.time()) ))
        print(np.sum(np.square(w_meta)))
        print("\n")
    if (i+1)%1000==0:
        accu_meta, accu_base = testProcedure()
        accu_record[0].append(accu_meta)
        accu_record[1].append(accu_base)


plt.plot(accu_record[0],'b-',label='meta')
plt.plot(accu_record[1],'r-',label='baseline')
plt.xlabel('training episode (x1000)')
plt.ylabel('accuracy')
#plt.ylim([0,10])
plt.title('Binary classification (k-shots) accuracy on MNIST, k = ' + str(k_shots))
plt.legend()
plt.savefig('../figure/'+str(k_shots)+'_shots_'+str(max_epsd)+'_epsd_'+str(num_task)+'_tasks_'+str(n_step_tuning)+'_steps.png')
plt.show()
