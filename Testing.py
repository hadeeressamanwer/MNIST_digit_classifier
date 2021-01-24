import forward_prop
import optimization
import evaluation
from init import X_test,Y_test,X_train,Y_train,X_batch,Y_batch,Y_evalution
import numpy as np
import utils_module


#layers_dims = [784,256,256,128,32,10]
layers_dims = [784,256,256,128,32,10]
X_train_t = np.transpose(X_train)
X_test_t = np.transpose(X_test)
X_batch_t = np.transpose(X_batch)
Y_test_t = np.transpose(Y_test)
Y_batch_t = np.transpose(Y_batch)
Y_train_t = np.transpose(Y_train)
X = X_batch_t
Y = Y_batch_t
# optimizer = "gd" , "adam" , "SGD" , "mini_batch" , "momentum"
optimizer = "adam"
# optimizer_mini_batch = "gd" , "adam" , "momentum"
optimizer_mini_batch = "adam"
# initialization = "random" , "zeros" ,"prev_parameters" -> initilaize with known values from parameters.npy file
initialization = "prev_parameters"
# ActivationFunctions = "relu" , "identity" , "sigmoid" , "tanh"
A_layers = "relu"
A_out = "relu"
num_iterations = 1000
learning_rate = 0.0005
print_cost = True
epsilon = 1e-8
mini_batch_size = 1000
beta1 = 0.9
beta2 = 0.999
beta = 0.9
# if prev_parameters not used make it else = utils_module.load()
prev_parameters = utils_module.load('parameters.py')
save = True
save_to_model = False

#if previous weights reaches good accuracy save it to model_parameters.py before saving the current weights to parameters.py
if save_to_model:
    model_parameters = utils_module.load('parameters.py')
    utils_module.save('model_parameters.py', model_parameters)


if optimizer == "gd":
    parameters = optimization.L_layer_model_GD(X, Y, layers_dims, initialization, A_layers, A_out,prev_parameters, learning_rate, num_iterations,
                     print_cost)
elif optimizer == "SGD":
    parameters = optimization.L_layer_model_SGD(X, Y, layers_dims, initialization, A_layers, A_out,prev_parameters, learning_rate, num_iterations, print_cost)
elif optimizer == "momentum":
    parameters = optimization.L_layer_model_GDWithMomentum(X, Y, layers_dims, initialization, A_layers, A_out,prev_parameters, beta, learning_rate, num_iterations, print_cost)

elif optimizer == "adam":
    parameters = optimization.L_layer_model_Adam(X, Y, layers_dims,initialization, A_layers , A_out ,prev_parameters,beta1 , beta2 ,  epsilon ,learning_rate,
                                                 num_iterations,  print_cost)

elif optimizer == "mini_batch":
    parameters = optimization.L_layer_model_minibatch(X, Y, layers_dims, optimizer_mini_batch, initialization, A_layers, A_out,prev_parameters,
                                                      mini_batch_size, learning_rate, beta,
                            beta1, beta2, epsilon, num_iterations, print_cost)


if save:
    utils_module.save('parameters.py',parameters)


Y_pred , cache = forward_prop.L_model_forward(X_test_t,parameters,A_layers,A_out)
Loss = forward_prop.compute_cost(Y_pred,Y_test_t)
Y_evalution_pred = np.argmax(Y_pred,axis=0)
evaluation.confusionmatrix(Y_evalution, Y_evalution_pred)



