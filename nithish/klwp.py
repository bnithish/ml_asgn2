import numpy as np
from utils import loadData
from sklearn.neighbors import NearestCentroid as lwp
from sklearn.neighbors import KNeighborsClassifier as knn
def doit(X,k):
    x,y = loadData("train",225)
    x = x.toarray()

    train_x = x[0:10000]
    train_y = y[0:10000]

    test_x = x[9000:10000]
    test_y = y[9000:10000]

    model = lwp()
    model.fit(train_x,train_y)
    prediction = model.predict(test_x)
    cent = model.centroids_
    clas = model.classes_
#  print(cent.shape)
#  print(clas)
    neigh = knn(n_neighbors=k)
    neigh.fit(cent,clas)
    kn = neigh.kneighbors(X.toarray())[:][1]
    #  correct = 0
    #  wrong = 0
    #  for i in range(1000):
        #  print(test_y[i],clas[kn[i]])
        #  if test_y[i] in clas[kn[i]]:
            #  correct = correct+1
        #  else:
            #  wrong = wrong+1
    #  print(correct,wrong)    
    return clas[kn]
