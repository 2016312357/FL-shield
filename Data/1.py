import numpy as np
import pandas as pd

'''# base = sample_generator.instance_a_train_loader(4, 32)
rs_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
train_data = pd.read_csv('./ua.base', sep='\t', names=rs_cols, encoding='utf-8')
user_ids = train_data["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = train_data["movie_id"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
train_data["user"] = train_data["user_id"].map(user2user_encoded)
train_data["movie"] = train_data["movie_id"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)
train_data["rating"] = train_data["rating"].values.astype(np.float32)
'''
'''
filename = "./ua.base"
ratingList = []
users=[]
with open('u.user', "r") as f:
    lines = f.readlines()
    for line in lines:

        arr = line.split("|")
        print(arr[2],arr[3])
        users.append([arr[2],arr[3]])
print(users)
with open(filename, "r") as f:
    line = f.readline()
    id=0
    while line is not None and line != "":

        arr = line.split("\t")
        user, item = int(arr[0]), int(arr[1])
        with open('./fed/'+str(user)+'_'+users[user-1][0]+'_'+users[user-1][1], "a") as w:
            w.write(line)

        line = f.readline()
    print('done')
'''
import os
name=[]
with open('../morethan200', "r") as ff:
    lines = ff.readlines()
    for line in lines:
        name.append(line[:-5])
        print(name)

with open('./train_data', "a") as w, open('./test_data', "a") as wt:
    for a in name:
        with open('./fed/' + a, "r") as f:
            lines = f.readlines()
            test=np.random.choice(len(lines), 1)
            for l in range(len(lines)):
                if l not in test:
                    w.write(lines[l])
                else:
                    wt.write(lines[l])







