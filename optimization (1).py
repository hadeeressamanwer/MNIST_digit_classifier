import numpy as np
from init import X_batch , Y_batch ,X_train ,Y_train
import init_param
import forward_prop
import back_prop
import visualization

def update_parameters_GD(parameters, grads, learning_rate):

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - (learning_rate * grads["dW" + str(l + 1)])
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - (learning_rate * grads["db" + str(l + 1)])

    return parameters


"""
layers_dims = row vector of nodes in each layer Ex.[3,2,2,9] inputFeatures = 3 & 3 layers 1st layer = 2.....
Initialization of parameters = ("he","random","zeros")
he intilization is recommended with Relu Activation Function
A_layers = Activation Function of Layers ("relu","sigmoid","identity")
A_out = A_layers = Activation Function of Output Layer ("relu","sigmoid","identity")
"""
def L_layer_model(X, Y, layers_dims,initialization, A_layers , A_out ,learning_rate=0.0075, num_iterations=1000,  print_cost=True):
    Y = np.transpose(Y)
    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization.
    if initialization == "zeros":
        parameters = init_param.initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters =init_param.initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = init_param.initialize_parameters_he(layers_dims)


    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: Activation["relu","sigmoid"]
        AL, caches = forward_prop.L_model_forward(X, parameters,A_layers,A_out)
        # Compute cost.
        cost =  forward_prop.compute_cost(AL, Y)
        # Backward propagation.
        grads = back_prop.L_model_backward(AL, Y, caches,A_layers,A_out)
        # Update parameters.
        parameters = update_parameters_GD(parameters, grads, learning_rate)



        # Print the cost every 100 iteration and i % 100 == 0
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    visualization.draw_costs(costs,learning_rate) 
    return parameters

#X = np.array([[6,2,3,4,5],
 #             [4,6,9,2,1]])
#Y = np.array([[1,0],[1,0],[0,1],[1,0],[1,0]])

#x2=np.array([[6],[4]])
#y2=np.array([[0,1]])
layersdim = [784,3,3,10]
X_batch_t=np.transpose(X_batch)
parameters = L_layer_model(X_batch_t,Y_batch,layersdim,"he","relu","identity")

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print("W3 = " + str(parameters["W3"]))
print("b3 = " + str(parameters["b3"]))
