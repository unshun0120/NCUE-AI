from sklearn.neighbors import NearestNeighbors
import numpy as np
import random

#宣告一個大小20*2的array
X = np.empty((20, 2))
#random
for i in range(20) : 
    X[i][0] = random.randint(-10, 10)
    X[i][1] = random.randint(-10, 10)

print("X array : ")
print(X)

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
print("indices : ")
print(indices)
print("distances : ")
print(distances)
print("kneighbors_graph")
print(nbrs.kneighbors_graph(X).toarray())
