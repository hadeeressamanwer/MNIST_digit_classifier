import forward_prop
import optimization
from init import X_test,Y_test,X_train,Y_train,X_batch,Y_batch
import numpy as np


layersdim = [784,3,3,3,10]

X_train_t = np.transpose(X_train)
X_test_t = np.transpose(X_test)
#X_batch_t = np.transpose(X_batch)
parameters = optimization.L_layer_model_minibatch(X_train_t,Y_train,layersdim,"adam","random","relu","identity",1000)
Y_pred , cache = forward_prop.L_model_forward(X_test_t,parameters,"relu","identity")
Y_test_t = np.transpose(Y_test)
Loss = forward_prop.compute_cost(Y_pred,Y_test_t)

print(Loss)


