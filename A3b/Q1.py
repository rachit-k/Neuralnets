import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys
import timeit
from sklearn.metrics import confusion_matrix

def onehot(Y):
    y=np.zeros((len(Y),np.amax(Y)+1))
    for i in range(len(Y)):
        y[i,Y[i]]=1
    return y

def sigmoid(X):
    return 1/(1 + np.exp(-1*X))

def relu(x):
    return x * (x > 0)

def loss(X,Y,num_layers,activation_fn,w,m):
    o = X.T
    if (activation_fn=="sigmoid"):
        for i in range(num_layers-1):
            o = sigmoid(np.matmul(w[i], o)) 
    else:
        for i in range(num_layers-1):
            o = relu(np.matmul(w[i], o)) 
    o = sigmoid(np.matmul(w[num_layers - 1], o)) 
    o = o.T
    return (np.sum((Y-o)**2)/(2*m))

def train(X,Y,batch_size,num_inputs,num_outputs,hidden_layer_list,activation_fn,learning_rate,learning_type):
    temp=X.shape
    m=temp[0]
#    n=temp[1]
    epoch=0
    num_layers=1+len(hidden_layer_list)
    nn_arch=[num_inputs]+hidden_layer_list+[num_outputs]
#    w=[((np.random.rand(nn_arch[i+1],nn_arch[i]))/5 - (np.random.rand(nn_arch[i+1],nn_arch[i]))/5) for i in range(num_layers)] 
    if(activation_fn=="sigmoid"): 
        w=[((np.random.rand(nn_arch[i+1],nn_arch[i]))*(math.sqrt(6/(nn_arch[i+1]+nn_arch[i]))) - math.sqrt(6/(nn_arch[i+1]+nn_arch[i]))) for i in range(num_layers)]
    else:
        w=[((np.random.rand(nn_arch[i+1],nn_arch[i]))/50 - 0.01) for i in range(num_layers)] 

#    w=np.array(w)
#    print(w[0].shape)
    new_loss=loss(X,Y,num_layers,activation_fn,w,m)
    old_loss=new_loss
    
    while(epoch<10000):        
        old_loss=new_loss
        if (m%batch_size == 0):
            num_batches = int(m/batch_size)
        else:
            num_batches = int((m/batch_size) + 1)
            
        for j in range(num_batches):
            x=X[j*batch_size:min(((j+1)*batch_size),m),:]
            y=Y[j*batch_size:min(((j+1)*batch_size),m),:]
            
            layers=[] #make into np arrays
            layers.append(x.T)
            if(activation_fn=="sigmoid"):
                for i in range(num_layers-1):
                    temp=sigmoid(np.matmul(w[i],layers[i])) 
                    layers.append(temp)
            else:
                for i in range(num_layers-1):
                    temp=relu(np.matmul(w[i],layers[i])) 
                    layers.append(temp)
            temp=sigmoid(np.matmul(w[num_layers-1],layers[num_layers-1])) 
            layers.append(temp)
            delj=[0 for i in range(num_layers)]
            delj[-1]=np.multiply((layers[num_layers] - y.T) , np.multiply(layers[num_layers],(1 - layers[num_layers])))
            for i in range(num_layers-2,-1,-1):
                if (activation_fn == 'sigmoid'):
                    delj[i] = np.multiply(np.matmul(w[i+1].T, delj[i+1]), np.multiply(layers[i+1],(1-layers[i+1]))) 
                else:
                    delj[i] = np.multiply(np.matmul(w[i+1].T, delj[i+1]), np.where(layers[i+1] > 0, 1, 0))  
            
            for i in range(num_layers-1, -1, -1):
                if (activation_fn == 'sigmoid'):
                    w[i] = w[i] - (learning_rate * np.matmul(delj[i], layers[i].T))
                else:
                    w[i] = w[i] - (learning_rate * np.matmul(delj[i], layers[i].T))/(2*batch_size)
                
        new_loss=loss(X,Y,num_layers,activation_fn,w,m) 
        if(epoch%100==0):
            print("epoch:",epoch)
            print(new_loss,old_loss)    
            
        if (epoch>200 and abs(new_loss - old_loss) < 0.0000001 and learning_type == "adaptive"):   
            print("convergence break at epoch:",epoch)
            break;
        if (epoch>200 and abs(new_loss - old_loss) < 0.000001 and learning_type == "constant"):
            print("convergence break at epoch:",epoch)
            break;
        if (learning_type == "adaptive"):
            if (epoch>0):
                learning_rate = learning_rate*(math.sqrt(epoch/(epoch+1)))
#                print(epoch,"changed learning rate to : ",learning_rate)    
        epoch=epoch+1    
    return w              
                    

def acc(prediction,real):
    accuracy =1.0 - (np.count_nonzero(real-prediction)/(len(prediction)))
    return accuracy*100       

def test(X,Y,num_layers,activation_fn,w):
    o = X.T
    if (activation_fn=="sigmoid"):
        for i in range(num_layers-1):
            o = sigmoid(np.matmul(w[i], o))
    else:
        for i in range(num_layers-1):
            o = relu(np.matmul(w[i], o))
    o = sigmoid(np.matmul(w[num_layers - 1], o))
    o=o.T
    prediction = np.argmax(o, axis = 1)
    real = np.argmax(Y, axis = 1)  
#    print(prediction)
#    print(real)
    return prediction,real


train_file = sys.argv[1]
test_file = sys.argv[2]
    
starttime = timeit.default_timer()  

data=pd.read_csv(train_file,header=None) #"train.csv"
data=np.array(data)

Y=data[:,-1] #(13000,)
#print(Y[0])
Y=onehot(Y)
#print(Y[0])
X=data[:,:-1] #(13000,784)
X=X/255.0

data=pd.read_csv(test_file,header=None) #test.csv
data=np.array(data)

tY=data[:,-1]
tY=onehot(tY)
tX=data[:,:-1]
tX=tX/255.0

print("reading time:")    
print(timeit.default_timer()-starttime)

starttime = timeit.default_timer() 

num_inputs=X.shape[1]
num_outputs=Y.shape[1]
batch_size=int(sys.argv[3]) #100
activation_fn=sys.argv[4] # relu sigmoid
learning_rate=float(sys.argv[5]) #0.1 0.5
learning_type=sys.argv[6] # adaptive constant

hidden_layer_list_temp=sys.argv[7] # [50]
hidden_layer_list=hidden_layer_list_temp.strip('][').split(', ') 
for i in range(0, len(hidden_layer_list)): 
    hidden_layer_list[i] = int(hidden_layer_list[i])

w=train(X,Y,batch_size,num_inputs,num_outputs,hidden_layer_list,activation_fn,learning_rate,learning_type)
num_layers=1+len(hidden_layer_list)
print("training time:")    
print(timeit.default_timer()-starttime)

prediction,real=test(X,Y,num_layers,activation_fn,w)
accuracy =acc(prediction,real)
print("train accuracy:")
print(accuracy)

prediction,real=test(tX,tY,num_layers,activation_fn,w)
accuracy =acc(prediction,real)
print("test accuracy:")
print(accuracy)

#layer_list=[[1],[5],[10],[50],[100]]
#acc_train_list=[]
#acc_test_list=[]
#time_list=[]
#
#for hidden_layer_list in layer_list:
#
#    print(hidden_layer_list)
#    w=train(X,Y,batch_size,num_inputs,num_outputs,hidden_layer_list,activation_fn,learning_rate,learning_type)
#    num_layers=1+len(hidden_layer_list)
#    
#    print("training time:")   
#    tim=timeit.default_timer()-starttime
#    print(tim)
#    time_list.append(tim)
#    
#    prediction,real=test(X,Y,num_layers,activation_fn,w)
#    accuracy =acc(prediction,real)
#    print("train accuracy:")
#    print(accuracy)
#    acc_train_list.append(accuracy)
#    
#    
#    prediction,real=test(tX,tY,num_layers,activation_fn,w)
#    accuracy =acc(prediction,real)
#    print("test accuracy:")
#    print(accuracy)
#    acc_test_list.append(accuracy)
#    
##cm = confusion_matrix(real, prediction)
##print(cm)
#
##
#plt.plot(layer_list,acc_train_list,'-o',label='train',color='red',)
#plt.plot(layer_list,acc_test_list,'-+',label='test',color='blue')
#plt.xlabel('hidden layer units')
#plt.ylabel('accuracy')
#plt.legend()
#plt.savefig("Q1cc.png",bbox_inches="tight")
#plt.show()
#
#plt.plot(layer_list,time_list,'-o')
#plt.xlabel('hidden layer units')
#plt.ylabel('time')
#plt.legend()
#plt.savefig("Q1dd.png",bbox_inches="tight")
#plt.show()

