from time import time
import os
import math
import random
import heapq
import numpy as np
from tensorflow import keras
from keras import backend as K

from tensorflow.keras import layers,models
from tensorflow.keras import initializers,regularizers,optimizers
from Dataset import Dataset1
from Client import Client
from train_model import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

class Single():
    def __init__(self, epochs = 100, verbose = 1,topK = 20,data_name = 'ml-1m', model_name = 'gmf'):
        self.epochs = epochs
        self.verbose = verbose
        self.topK = topK
        #dataset
        t1 = time()
        dataset = Dataset1("./Data/" + data_name)
        self.num_users, self.num_items = dataset.get_train_data_shape()
        self.test_datas = dataset.load_test_file()
        self.test_negatives = dataset.load_negative_file()
        self.train_datas = dataset.load_train_file()
        print("Server Load data done [%.1f s]. #user=%d, #item=%d, #test=%d"
          % (time()-t1, self.num_users, self.num_items, len(self.test_datas)))
        #model
        if model_name == "gmf":
            self.model = get_compiled_gmf_model(self.num_users,self.num_items)
        elif model_name == "mlp":
            self.model = get_compiled_mlp_model(self.num_users,self.num_items)
        elif model_name == "neumf":
            self.model = get_compiled_neumf_model(self.num_users,self.num_items)


    def get_weight_grad(self,model, inputs, outputs):
        """ Gets gradient of model for given inputs and outputs for all weights"""
        grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
        symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
        f = K.function(symb_inputs, grads)
        x, y, sample_weight = model._standardize_user_data(inputs, outputs)
        output_grad = f(x + y + sample_weight)
        return output_grad

    def get_layer_output_grad(self,model, inputs, outputs, layer=-1):
        """ Gets gradient a layer output for given inputs and outputs;
         returns the gradient at a given layer's output and there, the indexing is the same as in the model,
          so it's safe to use it.
         """
        grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
        symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
        f = K.function(symb_inputs, grads)
        x, y, sample_weight = model._standardize_user_data(inputs, outputs)
        output_grad = f(x + y + sample_weight)
        return output_grad



    def evaluate_model(self):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []
        for idx in range(len(self.test_datas)):
            rating = self.test_datas[idx]
            items = self.test_negatives[idx]
            user_id = rating[0]
            gtItem = rating[1]
            items.append(gtItem)
            # Get prediction scores
            map_item_score = {}
            users = np.full(len(items), user_id, dtype='int32')
            predictions = self.model.predict([users, np.array(items)],
                                        batch_size=100, verbose=0)
            for i in range(len(items)):
                item = items[i]
                map_item_score[item] = predictions[i]
            items.pop()
            # Evaluate top rank list
            ranklist = heapq.nlargest(self.topK, map_item_score, key=map_item_score.get)
            if gtItem in ranklist:
                hits.append(1)
                ndcgs.append(math.log(2)/math.log(ranklist.index(gtItem)+2))
            else:
                hits.append(0)
                ndcgs.append(0)
        return np.array(hits).mean(), np.array(ndcgs).mean()


    def run(self):
        t1 = time()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        self.model.compile(optimizer=optimizers.Adam(lr=0.001, clipnorm=5), loss='binary_crossentropy')
        self.model.summary()
        epoch=10
        self.model.set_weights(np.load('./checkpoints/' + str(epoch) + '.npz',allow_pickle=True)['arr_0'])
        print("##################",self.model.trainable_weights)

        hr, ndcg = self.evaluate_model()
        print(epoch, 'HR@20 = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))
        gr=[]
        id=[]

        for idx in range(2000):#len(self.test_datas)
            rating = self.test_datas[idx]
            user_id = rating[0]
            id.append(user_id)
            gtItem = rating[1]
            label = rating[2]
            label = [[label]]

            items = []#self.test_negatives[idx]
            items.append(gtItem)
            # Get prediction scores
            map_item_score = {}
            users = np.full(len(items), user_id, dtype='int32')

            '''outputTensor = self.model.output
            listOfVariableTensors = self.model.trainable_weights
            bce = tf.keras.losses.BinaryCrossentropy()
            loss = bce(outputTensor, label)
            gradients = K.gradients(loss, listOfVariableTensors)


            g = sess.run(gradients, feed_dict={self.model.input:[[users, np.array(items)]]})'''

            dummy_in = [users, np.array(items)]
            dummy_out = label
            dummy_loss = self.model.train_on_batch(dummy_in, dummy_out)
            #weight_grads = self.get_weight_grad(self.model, dummy_in, dummy_out)
            #for i in range(len(self.model.trainable_weights)):
            output_grad = self.get_layer_output_grad(self.model, dummy_in, dummy_out,layer=2)
            #print(output_grad)
            gr.append(output_grad)
            #print(np.asarray(gr).shape)
        np.savez('grad.npz',np.asarray(gr),np.asarray(id))






        # model是编译好的模型，
        '''with self.model.session.as_default():  # 模型所在的session
            with self.model.graph.as_default():  # 模型所在的graph
        '''
        # Train model federated
        '''best_hr, best_ndcg, best_iter = hr, ndcg, -1
        for epoch in range(self.epochs):
            t1 = time()
            hist = self.model.fit([np.array(self.train_datas[0]), np.array(self.train_datas[1])],  # input
                         np.array(self.train_datas[2]),  # labels
                         batch_size=64, epochs=1, verbose=0, shuffle=True)
        
            

            t2 = time()
            print('Iteration %d [%.1f s]'
              % (epoch,  t2-t1))

            if epoch % self.verbose == 0:
                hr, ndcg = self.evaluate_model()
                print('HR = %.4f, NDCG = %.4f [%.1f s]' % (hr, ndcg, time()-t2))
                if hr > best_hr:
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            np.savez('./checkpoints/' + str(epoch) + '.npz',self.model.get_weights())
        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))'''




S = Single(model_name="neumf")
S.run()