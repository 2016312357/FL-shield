import copy
from time import time
import os
import math
import random
import heapq
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import initializers, regularizers, optimizers
from Dataset import Dataset
from Client import Client

from Privacy import Privacy_account
from train_model import *  # define model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'


# Server:distribute_task & evaluate
class Server:
    def __init__(self, epochs=20, verbose=1, topK=20, lr=0.01, cilp_norm=0.5, data_name='ml-1m', model_name='neumf'):
        self.epochs = epochs
        self.verbose = verbose
        self.topK = topK
        self.C = cilp_norm
        self.lr = lr
        # dataset
        t1 = time()
        dataset = Dataset("./Data/")
        self.num_users, self.num_items = dataset.get_train_data_shape()  # all users
        self.test_datas = dataset.load_test_file()
        self.test_negatives = dataset.load_negative_file()
        print("Server Load data done [%.1f s]. #user=%d, #item=%d, #test=%d"
              % (time() - t1, self.num_users, self.num_items, len(self.test_datas)))
        # model
        if model_name == "gmf":
            self.model = get_compiled_gmf_model(self.num_users, self.num_items)
        elif model_name == "mlp":
            self.model = get_compiled_mlp_model(self.num_users, self.num_items)
        elif model_name == "neumf":
            self.model = get_compiled_neumf_model(self.num_users, self.num_items)
        # init clients
        self.client = Client()

    def distribute_task(self, epoch, client_ids, budget=100, layer=0, gc=0, dp=10):  # get clients train parameters
        server_weights = self.model.get_weights()  # initial global model
        client_weight_datas = []
        # w=[]#print('this epoch',client_ids)
        for client_id in client_ids:
            # return model updates
            weights = self.client.train_epoch(self.model, client_id, server_weights,
                                              epoch, budget=budget, layer=layer, gc=gc)
            client_weight_datas.append(weights)
            # w.append(weights)#[-4].reshape((1,1280)))
        # print(client_weight_datas[0][-4])

        # np.savez('./checkpoints-/' + str(epoch) + '.npz', np.vstack(w), np.array(client_ids))
        return client_weight_datas, client_ids

    def federated_average(self, client_weight_datas, init_weights, noise_scale, dp=False):
        client_num = len(client_weight_datas)
        assert client_num != 0
        if dp:
            print('add noise')
            for k in range(client_num):
                w_noise = copy.deepcopy(client_weight_datas[k])
                for i in range(len(w_noise)):
                    noise = np.random.normal(0, noise_scale, w_noise[i].shape)
                    client_weight_datas[k][i] = w_noise[i] + noise

        w = client_weight_datas[0]
        for i in range(1, client_num):
            w += client_weight_datas[i]
        # print(init_weights)
        w = w / client_num + init_weights  # do gradient average
        # self.model.set_weights(w)  # update weight
        #print('agreeeeeeeeeeeeeeeeeeeeeeeee', init_weights)
        return w

    '''def getHitRatio(ranklist, gtItem):#是否hit
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    # NDCG
    def getNDCG(ranklist, gtItem):#hit的程度
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return np.log(2) / np.log(i + 2)#归一化折损累计增益
    '''

    def evaluate_model(self):
        """
        output results of 1-k
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [[] for _ in range(self.topK)], [[] for _ in range(self.topK)]
        for idx in range(len(self.test_datas)):
            rating = self.test_datas[idx]
            items = self.test_negatives[idx]
            # items=[]
            user_id = rating[0]
            gtItem = rating[1]  # test movie
            items.append(gtItem)
            # Get prediction scores
            map_item_score = {}
            users = np.full(len(items), user_id, dtype='int32')
            # print('hihihi',users,users.shape)#(100,)
            # np.full 构造一个数组，用指定值填充其元素
            # full(shape, fill_value, dtype=None, order='C')shape：int 或者 int元组
            # fill_value：填充到数组中的值
            predictions = self.model.predict([users, np.array(items)],
                                             batch_size=100, verbose=0)
            # print(predictions)
            for i in range(len(items)):
                item = items[i]
                map_item_score[item] = predictions[i]
            items.pop()
            # Evaluate top rank list
            ranklist = heapq.nlargest(self.topK, map_item_score, key=map_item_score.get)

            if gtItem in ranklist:  # hit
                p = ranklist.index(gtItem)
                for i in range(p):
                    hits[i].append(0)
                    ndcgs[i].append(0)
                for i in range(p, self.topK):
                    hits[i].append(1)
                    ndcgs[i].append(math.log(2) / math.log(ranklist.index(gtItem) + 2))
            else:  # no hit
                for i in range(self.topK):
                    hits[i].append(0)
                    ndcgs[i].append(0)
        hits = [np.array(hits[i]).mean() for i in range(self.topK)]
        ndcgs = [np.array(ndcgs[i]).mean() for i in range(self.topK)]
        return hits, ndcgs

    def run(self):
        user = []
        with open('morethan200', 'r') as f:
            lines = f.readlines()
            for l in lines:
                user.append(int(l.split('_')[0]))
        print('total users', len(user), user)

        N = 500
        # layerid = [0, 1, 2, 3]
        # [1,0,3,2,4,6]
        layerid = [1]  # higher than 0.92
        ll = ''
        for l in layerid:
            ll += str(l) + '_'

        try:
            self.model.set_weights(np.load('global_model_weights_layer_{}epoch-{}-dp10.npz'.format(ll, N))['x'])
            # np.savez('ml-10k-layer_{}epoch-{}.npz'.format(ll, N), w, idss)

            print('loading model')
        except:
            print('initializing')

            self.model.summary()

        hrs, ndcgs = self.evaluate_model()
        for i in range(self.topK):
            print('HR@%d = %.4f, NDCG@%d = %.4f' % (i + 1, hrs[i], i + 1, ndcgs[i]))
        # print('initial test using [%.1f s]' % (time() - t1))

        # self.model.summary()
        w = []
        idss = []

        threshold_epochs = copy.deepcopy(self.epochs)
        noise_list = []
        epsilon=5
        noise_scale = copy.deepcopy(Privacy_account(threshold_epochs, noise_list, 0, clipthr=5,
                                                    delta=0.001, privacy_budget=epsilon))
        print('privacy budget',epsilon,'noise scale:',noise_scale)
        # Privacy_account(threshold_epochs, noise_list, iter, delta, privacy_budget, dp_mechanism='Origi'):
        self.model.compile(optimizer=optimizers.Adam(lr=self.lr, clipnorm=5), loss='binary_crossentropy')

        for epoch in range(self.epochs):
            # t1 = time()

            for i in range(1):  # 每个客户端每次分配任务时只进行一次训练，因此需要重复多次分配
                server_weights = copy.deepcopy(self.model.get_weights())
                #print(epoch, server_weights)
                # print(epoch,server_weights)
                client_weight_datas, ids = self.distribute_task(epoch + 1,
                                                                random.sample(user, 10), budget=N,
                                                                layer=layerid, gc=0)

                client_weights = self.federated_average(client_weight_datas, server_weights, noise_scale,
                                                        dp=True)  # do Average
                w.append(client_weight_datas)
                idss.append(ids)
                # self.C = 10*self.lr*max([np.linalg.norm(client_weights[la] - server_weights[la]) for la in range(len(client_weights))])
                # self.model.compile(optimizer=optimizers.Adam(lr=self.lr, clipnorm=self.C), loss='binary_crossentropy')
                # print(self.C)
                self.model.set_weights(client_weights)

            '''if (epoch+1)%5==0:
                self.lr*=0.1
                print(self.lr)
                self.model.compile(optimizer=optimizers.Adam(lr=self.lr, clipnorm=5),
                                   loss='binary_crossentropy')'''

            # np.savez('ml-100k-lap-5.npz'.format(ll, N), w, idss)

            # t2 = time()
            # print('Iteration %d [%.1f s]' % (epoch,  t2-t1))

            if epoch % self.verbose == 0:
                hrs, ndcgs = self.evaluate_model()
                for i in range(self.topK - 1, self.topK):
                    with open('./acc_layer_{}epoch{}-dp5.txt'.format(ll, N), 'a+') as f:
                        # f.write('HR@%d = %.4f, NDCG@%d = %.4f\n' % (i + 1, hrs[i], i + 1, ndcgs[i]))
                        f.write('HR@%d = %.4f \n' % (i+1,hrs[i]))
                    print('HR@%d = %.4f, NDCG@%d = %.4f' % (i + 1, hrs[i], i + 1, ndcgs[i]))

                # print(self.verbose, 'epochs totally consume [%.1f s]' % (time() - t1))
                # np.savez('global_model_weights_layer_{}epoch-{}.npz'.format(ll, N), x=self.model.get_weights())
            np.savez('ml-100k-layer_{}epoch-{}-dp5.npz'.format(ll, N), w, idss)
            print('saving done')

        '''
                if hr > best_hr:
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch
        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
        '''


ser = Server(epochs=30, verbose=1)  # 每verbose轮输出一次
ser.run()
