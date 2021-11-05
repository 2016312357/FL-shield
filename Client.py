from time import time
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import initializers, regularizers, optimizers
from Dataset import Dataset
# from main import generate_adv
# from main import generate_adv

from train_model import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def sparse_top_k(x, input_compress_settings={}):
    compress_settings = {'k': 0.3}  # prune去掉k%的梯度。
    compress_settings.update(input_compress_settings)
    k = compress_settings['k']
    k = 1 - k
    vec_x = x.flatten()
    d = int(len(vec_x))
    # print(d)
    k = int(np.ceil(d * k))
    # print(k)
    indices = np.argsort(np.abs(vec_x))[-1 * k:]  # max k  .topk(k)[1]
    out_x = np.zeros_like(vec_x)
    out_x[indices] = vec_x[indices]
    out_x = out_x.reshape(x.shape)
    # print(x.shape)
    return out_x


class Client:
    def __init__(self, batch_size=64, epochs=1,
                 data_name='ml-1m', model_name='gmf'):
        self.epochs = epochs
        self.batch_size = batch_size
        # get dataset
        t1 = time()
        dataset = Dataset("./Data/")
        self.num_users, self.num_items = dataset.get_train_data_shape()
        self.client_train_datas = dataset.load_client_train_date()
        print("Client Load data done [%.1f s]. #user=%d, #item=%d"
              % (time() - t1, self.num_users, self.num_items))

    def train_epoch(self, server_model, client_id, server_weights, epoch, budget=100, layer=0, gc=0, dp=False):
        train_data = self.client_train_datas[client_id]
        init_weight = server_weights  # np.array(server_model.get_weights())
        server_model.set_weights(server_weights)

        # print(server_weights)

        # print( init_weight)
        # print(self.batch_size)
        server_model.fit([np.array(train_data[0]), np.array(train_data[1])],  # input
                         np.array(train_data[2]),  # labels
                         batch_size=32, epochs=5, verbose=0, shuffle=True)  # train 20 epochs

        grads = np.array(server_model.get_weights())  # Model updates
        #print('train client ', grads == init_weight)
        grads -= init_weight

        # gc iclr18
        if gc > 0:
            for i in range(len(init_weight)):
                grads[i] = sparse_top_k(grads[i], input_compress_settings={'k': gc})
        # if dp:

        # cw2017, my defense
        '''if budget > 0:
            grads = generate_adv(grads, client_id=client_id,
                                 k=np.random.choice(range(21), 1)[0],
                                 max_epoch=budget, layerids=layer)'''

        '''
        # add noise
        epsilon = 0.5
        delta = 0.00001
        sensitivity = 0.001/64 * math.sqrt(2 * math.log(1.25/delta))/epsilon
        sigma = sensitivity/epsilon * math.sqrt(2 * math.log(1.25/delta))
        # noise = np.random.normal(0, sigma)
        noise = np.random.normal(0, sigma/math.sqrt(5), init_weight.shape)'''

        '''
        #分层加入噪声
        epsilon = 10
        for i in range(len(init_weight)):
            increment = grads[i] 
            #print("weights:")
            #print(weights[i])
            #print("delta:")
            #print(increment)
            sensitivity = increment.max() - increment.min()
            #print(sensitivity)
            sigma = sensitivity/epsilon#ss *
            noise = np.random.laplace(0, sigma, init_weight[i].shape)

            #noise = np.random.normal(0, sigma, init_weight[i].shape)
            grads[i] += noise
        '''

        return grads

        '''
        for i in range(int(len(train_data[0])/self.batch_size)):  # batch num
            gradients = None
            cur_weights = np.array(server_model.get_weights())
            begin = i * self.batch_size
            #end = begin
            end=begin+self.batch_size
            if begin + self.batch_size > len(train_data[0]):
                end = len(train_data[0])
            server_model.set_weights(cur_weights)
            server_model.tran_on_batch(
                [np.array(train_data[0][begin:end]), np.array(train_data[1][begin:end])],  # input
                np.array(train_data[2][begin:end]),  # labels
                batch_size=self.batch_size, epochs=self.epochs, verbose=0, shuffle=True)  # train 20 epoch'''

        # print('train epoch:',self.epochs)#1

        '''
        for i in range(int(len(train_data[0]) / self.batch_size)):  # batch num
            gradients = None
            cur_weights = np.array(server_model.get_weights())
            begin = i * self.batch_size
            end = begin
            for j in range(self.batch_size):
                if begin + j >= len(train_data[0]):
                    break;
                server_model.set_weights(cur_weights)
                server_model.fit([np.array(train_data[0][begin + j:begin + j + 1]),
                                  np.array(train_data[1][begin + j:begin + j + 1])],  # input
                                 np.array(train_data[2][begin + j:begin + j + 1]),  # labels
                                 batch_size=1, epochs=self.epochs, verbose=0, shuffle=True)  # train 1 epoch
                end += 1
                if j != 0:
                    gradients += np.array(server_model.get_weights())
                else:
                    gradients = np.array(server_model.get_weights())
                    # （[-0.09109745，-0.09036621、0.0977743，-0.07977977、0.10829113]，dtype = float32），

            server_model.set_weights(gradients / (end - begin))  # average
            # print('train for batch: ', end-begin)#64

        weights = np.array(server_model.get_weights())
        grads = weights - init_weight  # Model updates
        #grads = generate_adv(grads, client_id=client_id, k=np.random.choice(range(21), 1)[0])
        # server_model.set_weights(weights)'''
        '''np.savez('./checkpoints-2layer-gradients/' + str(client_id) + '_' + str(epoch) + '.npz',
                 grads)'''

        '''#add noise
        epsilon = 0.5
        delta = 0.00001
        sensitivity = 0.001/64 * math.sqrt(2 * math.log(1.25/delta))/epsilon
        sigma = sensitivity/epsilon * math.sqrt(2 * math.log(1.25/delta))
        #noise = np.random.normal(0, sigma)
        noise = np.random.normal(0, sigma/math.sqrt(5), weights.shape)
        
        #分层加入噪声
        for i in range(len(weights)):
            increment = weights[i] - server_weights[i]
            print("weights:")
            print(weights[i])
            print("delta:")
            print(increment)
            sensitivity = increment.max() - increment.min() 
            sigma = ss * sensitivity
            noise = np.random.normal(0, sigma, weights[i].shape)
            weights[i] += noise
        '''
