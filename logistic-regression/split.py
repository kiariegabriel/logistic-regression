import numpy as np
from random import sample


def split(X,t,p=0.25):
    length=X.shape[0]
    test_size=int(p*length)
    train_size=length-test_size
    train_indices=sample(range(length),train_size)
    test_indices=[]
    for i in range(length):
        if i not in train_indices:
            test_indices.append(i)
    
    X_train_list=[]
    y_train_list=[]
    for i in train_indices:
        row_list=[]
        y_train_list.append(t[i])
        row=X[i,:]
        for i in row:
            row_list.append(i)
        X_train_list.append(row_list)
    X_train=np.array(X_train_list)
    y_train=np.array(y_train_list)
        
    X_test_list=[]
    y_test_list=[]
    for i in test_indices:
        row_list=[]
        y_test_list.append(t[i])
        row=X[i,:]
        for i in row:
            row_list.append(i)
        X_test_list.append(row_list)
    
    X_test=np.array(X_test_list)
    y_test=np.array(y_test_list)
    #print(X_train.shape,X_test.shape)
    #print(y_train.shape,y_test.shape)
    return X_train,y_train,X_test,y_test