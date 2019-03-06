
import numpy as np


def sigmoid(x):
    # support matrix
    return 1.0/(1+np.exp(-x))

def learnRate(t, learn_rate_ini, T):
    return learn_rate_ini/(1+t/T)

def accuracyLogi(t,y):
    # support matrix
    p = np.where(y<0.50, 1, 0)
    return np.mean(t^p)



# Compute first order approximation gradient of theta_prime for a single task

def getGradient(inputs_a, targets_a, inputs_b, targets_b, w, k_shots, learn_rate_alpha, is_train):

    # support set
    # compute gradient
    x = np.concatenate((inputs_a[:k_shots], inputs_b[:k_shots]), axis=0)
    t = np.concatenate((targets_a[:k_shots], targets_b[:k_shots]), axis=0)
    y = sigmoid(np.dot(x,w))    
    loss_train = (np.dot(t.transpose(), np.log(y)) + np.dot(1-t.transpose(), np.log(1-y))).item()
    accu_train = accuracyLogi(t,y)
    gw = np.dot(x.transpose(),t-y) / k_shots    # negative gradient
    
    w_star = w + learn_rate_alpha*gw    # calculate updated parameter

    # query set 
    # use updated w_star as parameter to evaluate and compute gradient
    x = np.concatenate((inputs_a[k_shots:], inputs_b[k_shots:]), axis=0)
    t = np.concatenate((targets_a[k_shots:], targets_b[k_shots:]), axis=0)
    y = sigmoid(np.dot(x,w_star))    
    loss_query = (np.dot(t.transpose(), np.log(y)) + np.dot(1-t.transpose(), np.log(1-y))).item()
    accu_query = accuracyLogi(t,y)

    gw_star = np.dot(x.transpose(),t-y) / k_shots    # negative gradient
    gradient = gw_star if is_train else gw
        
    return gradient, accu_train, accu_query



# Sample 1) batch of tasks; 2) k-shots data points

def getSamples(dataLoaders, ctgy_range, sample_size):
    
    # sample two categories randomly, then sample k-shots data points inside
    ctgy_samples = np.random.choice(ctgy_range, 2, replace=False)
    ctgy_alpha = ctgy_samples[0] if ctgy_samples[0] < ctgy_samples[1] else ctgy_samples[1]
    ctgy_bravo = ctgy_samples[1] if ctgy_samples[0] < ctgy_samples[1] else ctgy_samples[0]
    dataLoaders[ctgy_alpha].reset(sample_size)
    dataLoaders[ctgy_bravo].reset(sample_size)
    inputs_alpha, targets_alpha, _ = dataLoaders[ctgy_alpha].getMiniBatch()
    inputs_bravo, targets_bravo, _ = dataLoaders[ctgy_bravo].getMiniBatch()
    targets_alpha = np.where(targets_alpha>0, 0, 1)
    targets_bravo = np.where(targets_bravo>0, 1, 0)
    
    return_data = (inputs_alpha, targets_alpha, inputs_bravo, targets_bravo)
    return return_data

