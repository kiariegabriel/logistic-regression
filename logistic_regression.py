import numpy as np




def sigmoid(a):
    return 1/(1+np.exp(-a))

def loss_function(X,t,w):
    loss_list=[]
    eps=1e-8
    for indx in range(X.shape[0]):
        a=np.dot(w.T,X[indx,:])
        y=sigmoid(a)
        loss=-((t[indx]*np.log(y+eps))+((1-t[indx])*np.log(1-y+eps)))
        loss_list.append(loss)
    mean=sum(loss_list)/X.shape[0]
    
    return mean

def gradient_function(X,t,w):
    gradient_list=[]
    for indx in range(X.shape[0]):
        a=np.dot(w.T,X[indx,:])
        y=sigmoid(a)
        gradient=(y-t[indx])*X[indx,:]
        gradient_list.append(gradient)
    avg_gradient=sum(gradient_list)/X.shape[0]
    
    return avg_gradient


def fit(X_train,y_train,num_iter=1000,learning_rate=0.01):
    losses=[]
    w=np.random.rand(64)

    for i in range(num_iter):
        losses.append(loss_function(X_train,y_train,w))
        w=w-0.01*gradient_function(X_train,y_train,w)
    return losses,w

def predict(X_test,w):
    predicted=[]
    for indx in range (X_test.shape[0]):
        predicted_value=sigmoid(np.dot(w.T,X_test[indx,:]))
        if predicted_value<=0.05:
            predicted_value=0
        elif predicted_value>=0.95:
            predicted_value=1
        predicted.append(predicted_value)
    return predicted

def accuracy(predicted,y_test):
    count=0
    for indx in range(len(predicted)):
        if predicted[indx]!=y_test[indx]:
            count+=1
    acc=(len(y_test)-count)/len(y_test)
    return acc