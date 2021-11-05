import random
import numpy as np
a=np.mean([1,3,5,6,7,8])
print(a)
import scipy.sparse as sp
import collections

def load_client_train_date():  # for FL train

    filename = "./Data/train_data"
    num_users = 0
    num_items = 0
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
    num_items += 1
    num_users += 1
    print(num_users,num_items)

    mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            # print("usr:{} item:{} score:{}".format(user,item,rating))
            if rating > 0:
                mat[user, item] = 1.0
            line = f.readline()

    #client_datas = [[[], [], []] for i in range(num_users)]  # 三元组 !!!!!!!!!!!!!!
    with open('./Data/test_negative','a+') as w, open('./Data/test_data','r') as r:
        line = r.readline()
        while line is not None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            w.write('('+str(user)+','+str(item)+')')
            line = r.readline()

            for t in range(99):  # 任意选择一个没有看的电影，作为negative项
                nega_item = np.random.randint(num_items)
                while nega_item==item or (user, nega_item) in mat.keys():
                    nega_item = np.random.randint(num_items)
                w.write('\t'+str(nega_item))
            w.write('\n')

    print('done')
#load_client_train_date()