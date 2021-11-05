# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:37:06 2019

@author: WEIKANG
"""

import numpy as np
import copy

import math



def Privacy_account(threshold_epochs, noise_list, iter,clipthr,delta,privacy_budget,dp_mechanism = 'Origi'):
    q_s = 10/100
    delta_s = 2*clipthr/200#args.clipthr/args.num_items_train
    if dp_mechanism != 'CRD':
        noise_scale = delta_s*np.sqrt(2*q_s*threshold_epochs*np.log(1/delta))/privacy_budget
    elif dp_mechanism == 'CRD':  # noise scale 随训练轮数动态变化
        noise_sum = 0
        for i in range(len(noise_list)):
            noise_sum += pow(1/noise_list[i],2)
        if pow(privacy_budget/delta_s,2)/(2*q_s*np.log(1/delta))>noise_sum:
            noise_scale = np.sqrt((threshold_epochs-iter)/(pow(privacy_budget/delta_s,2)/(2*q_s*np.log(1/delta))-noise_sum))
        else:
            noise_scale = noise_list[-1]
    return noise_scale


def Adjust_T(args, loss_avg_list, threshold_epochs_list, iter):
    if loss_avg_list[iter-1]-loss_avg_list[iter-2]>=0:
        threshold_epochs = copy.deepcopy(math.floor( math.ceil(args.dec_cons*threshold_epochs_list[-1])))
        # print('\nThreshold epochs:', threshold_epochs_list)
    else:
        threshold_epochs = threshold_epochs_list[-1]
    return threshold_epochs

