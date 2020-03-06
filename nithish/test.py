import numpy as np
from utils import loadData
from sklearn.tree import DecisionTreeClassifier as dt
def doit(inp,k):
    x,y = loadData("train",225)
    x = x.toarray()
    train_x = x[0:10000]
    train_y = y[0:10000]

    test_x = x[9000:10000]
    test_y = y[9000:10000]

    model = dt(max_depth=10)
    model.fit(train_x,train_y)
    ret = model.predict_proba(X=inp)
    predict = model.predict(X=inp)
    clas = model.classes_
    for i in range(ret.shape[0]):
        ret[i] = clas[np.argsort(ret[i])]
        #  print(predict[i],ret[i][ret[0].size-1:ret[0].size])    
    return np.flip(ret[:,ret[0].size-k:ret[0].size],axis = 1)
