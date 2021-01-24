import cnn 
import math
import init 
import numpy as np
def lnet(X_train): # 60000x784 >> 60000 28 28 1
    first=int(math.sqrt(X_train.shape[1]))
    X_train=X_train.reshape(X_train.shape[0],first,first,1)
    #conv layer1 
    w1=np.matrix([[1]])
    w1 =np.reshape(w1,(1,4,4,1))
    print (w1.shape)
    b1=np.random.randint(0,(1,1,1,6))
    #A1=cnn.conv(X_train,w1,b1,1,0)
    #pooling layer 1
    #p1=cnn.pool(A1,2,2,"average")
    #5x5 p=0 s=1 16 filter conv layer 2
    w2=np.random.randint(5,5,6,16)
    b2=np.random.randint(1,1,1,16)
    #A2=cnn.conv(p1,w2,b2,1,0)
    #pooling layer 2
    #p2=cnn.pool(A2,2,2,"average")
    #flatten
    input_fc=p2.reshape(60000,4*4*16)
    layers_dims = [input_fc.shape[1],120,84,10]


lnet(init.X_train)
    


    
