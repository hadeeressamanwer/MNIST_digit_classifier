# MNIST_digit_classifier
implement deep learning model to classify digits

## Installation:
### pip install phoenix_ml5
### from phoenix_m15 import Framework


## Main modules:
* Back Propagation.
* Forward Propagation.
* Data Loader.
* Evaluation.
* Classifier.
* NN.
* Optamizer.
* Visualization.
* utilitis Module.
* Data Preprocessing module for handing nan and onehotencoding.
* CNN Forward propagation.


##steps
step1 :
import dependencies

steps 2 :
preparing dataset
-use download_mnist() function to download mnist dataset
-use save_mnist() function to save mnist in mnist.pkl

steps 3 :
use load function () to 
load data for training and testing from mnist.pkl
return X_train, Y_train, X_test, Y_test 

step 4 :
use MakeOneHot( Y, D_out)
take 2 parameters
label data (Y_train OR Y_test) 
D_out number of classes (10 for mnist data )
to specify the correct class
return matrix containing all examples for each example zeroing all the values except the correct class

step 4 :
initialize parameters  
choose initialization method ("random" , "zeros" )
or use pretrained model from parameters.py so set initialization to "prev_parameters"

step 5 :
training
choose optimization method 
gd,SGD,momentum,adam

if gd (gradient descent is selected use this fn)
use this fn
L_layer_model_GD(X, Y, layers_dims, initialization, A_layers, A_out,prev_parameters, learning_rate, num_iterations,print_cost)
X->trainig (x_train)
Y->y label output from onehot function
layers_dims-> (number of input features,no.of nodes in 1st layer,no.of nodes in 2nd layer,no.of nodes in 3rd layer,.....,no.of nodes in m layer)
A_layers-> activation function for all layers except last layer provided options ("relu","sigmoid","identity")
A_out->activation function for last layer provided options ("relu","sigmoid","identity")
prev_parameters -> is used to save model parameters 
this function returns parameters after training


if sgd (stochastic gradient descent is selected use this fn)
use
 L_layer_model_SGD(X, Y, layers_dims, initialization, A_layers, A_out,prev_parameters, learning_rate, num_iterations, print_cost)
X->trainig (x_train)
Y->y label output from onehot function
layers_dims-> (number of input features,no.of nodes in 1st layer,no.of nodes in 2nd layer,no.of nodes in 3rd layer,.....,no.of nodes in m layer)
A_layers-> activation function for all layers except last layer provided options ("relu","sigmoid","identity")
A_out->activation function for last layer provided options ("relu","sigmoid","identity")
prev_parameters -> is used to save model parameters 
this function returns parameters after training 


if momentum
use
 L_layer_model_GDWithMomentum(X, Y, layers_dims, initialization, A_layers, A_out,prev_parameters, beta, learning_rate, num_iterations, print_cost)
X->trainig (x_train)
Y->y label output from onehot function
layers_dims-> (number of input features,no.of nodes in 1st layer,no.of nodes in 2nd layer,no.of nodes in 3rd layer,.....,no.of nodes in m layer)
A_layers-> activation function for all layers except last layer provided options ("relu","sigmoid","identity")
A_out->activation function for last layer provided options ("relu","sigmoid","identity")
prev_parameters -> is used to save model parameters 
this function returns parameters after training 
beta ->choose momentum paramete (0-1)
this function returns parameters after training 

if adam
use
L_layer_model_Adam(X, Y, layers_dims,initialization, A_layers , A_out ,prev_parameters,beta1 , beta2 ,  epsilon ,learning_rate,
                                                 num_iterations,  print_cost)
X->trainig (x_train)
Y->y label output from onehot function
layers_dims-> (number of input features,no.of nodes in 1st layer,no.of nodes in 2nd layer,no.of nodes in 3rd layer,.....,no.of nodes in m layer)
A_layers-> activation function for all layers except last layer provided options ("relu","sigmoid","identity")
A_out->activation function for last layer provided options ("relu","sigmoid","identity")
prev_parameters -> is used to save model parameters 
this function returns parameters after training 
beta 1 ->choose parameter (0-1)
beta 2 ->choose parameter (0-1)
this function returns parameters after training 

comment :
for using minibatch trainig 
use this function
L_layer_model_minibatch(X, Y, layers_dims, optimizer_mini_batch, initialization, A_layers, A_out,prev_parameters,
                                                      mini_batch_size, learning_rate, beta,
                            beta1, beta2, epsilon, num_iterations, print_cost)
this function returns parameters after training 

step 6:
testing 
use 
L_model_forward(input,parameters,A_layers,A_out)


step 7: 
model evaluation
to draw confusion matrix 
call this function
confusionmatrix(Y_evalution, Y_evalution_pred)
this function returns confusion matrix : accuracy , Precision, Recall, F1 score for each class 



