import numpy as np
import pandas as pd
import sys
import timeit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def onehot(Y):
    y=np.zeros((len(Y),np.amax(Y)+1))
    for i in range(len(Y)):
        y[i,Y[i]]=1
    return y

train_file = sys.argv[1]
test_file = sys.argv[2]

starttime = timeit.default_timer()  

data=pd.read_csv(train_file,header=None)
data=np.array(data)

Y=data[:,-1]
Y=onehot(Y)
X=data[:,:-1]
X=(X/255.0) 
data=pd.read_csv(test_file,header=None)
data=np.array(data)

tY=data[:,-1]
tY=onehot(tY)
tX=data[:,:-1]
tX=(tX/255.0)
print("reading time:")    
print(timeit.default_timer()-starttime)

starttime = timeit.default_timer()     

#mlp = MLPClassifier(hidden_layer_sizes=(100,100),activation = 'relu',solver='sgd',learning_rate='invscaling',learning_rate_init=0.5,power_t=0.5)
mlp = MLPClassifier(hidden_layer_sizes=(100,100),activation = 'relu',solver='sgd',learning_rate='constant',learning_rate_init=0.1)
mlp.fit(X,Y)

print("training time:")    
print(timeit.default_timer()-starttime)

#y_pred = mlp.predict(tX)
#cm = confusion_matrix(y_pred, tY)
#print(cm)

#prob=mlp.predict_proba(X)
#y_pred=np.argmax(prob, axis = 1)
print("train accuracy:")
#accuracy =1.0 - (np.count_nonzero(Y-y_pred)/(len(Y)))
#print(accuracy)
accuracy = mlp.score(X,Y)
print(accuracy)

#prob=mlp.predict_proba(tX)
#y_pred=np.argmax(prob, axis = 1)
print("test accuracy:")  
#accuracy =1.0 - (np.count_nonzero(tY-y_pred)/(len(tY)))
#print(accuracy)
accuracy = mlp.score(tX,tY)
print(accuracy)