implement deep learning model to classify digits

## Installation:
### pip install phoenix_ml9
### from phoenix_m19 import Framework


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

## dependencies<br/>
import pickle<br/>
import numpy as np<br/>
from urllib import request<br/>
import gzip<br/>
import math<br/>
#for visualization<br/>
import numpy as np<br/>
import pandas as pd<br/>
import seaborn as sn<br/>
import matplotlib.pyplot as plt<br/>
from prettytable import PrettyTable<br/>

## steps
## step1 :<br/> 
import dependencies<br/>

## steps 2 :<br/>
preparing dataset<br/>
-use download_mnist() function to download mnist dataset<br/> 
-use save_mnist() function to save mnist in mnist.pkl<br/>

## steps 3 :<br/>
use load function () to<br/>
load data for training and testing from mnist.pkl<br/>
return X_train, Y_train, X_test, Y_test<br/>

## step 4 :<br/>
use MakeOneHot( Y, D_out)<br/>
take 2 parameters<br/>
label data (Y_train OR Y_test)<br/>
D_out number of classes (10 for mnist data )<br/>
to specify the correct class<br/>
return matrix containing all examples for each example zeroing all the values except the correct class<br/>

## step 4 :<br/>
initialize parameters<br/>
choose initialization method ("random" , "zeros" )<br/>
or use pretrained model from parameters.py so set initialization to "prev_parameters"<br/>

## step 5 :<br/>
training<br/>
choose optimization method<br/>
gd,SGD,momentum,adam<br/>

if gd (gradient descent is selected use this fn)<br/>
use this fn<br/>
L_layer_model_GD(X, Y, layers_dims, initialization, A_layers, A_out,prev_parameters, learning_rate, num_iterations,print_cost,print_every)<br/>
X->trainig (x_train)<br/>
Y->y label output from onehot function<br/>
layers_dims-> (number of input features,no.of nodes in 1st layer,no.of nodes in 2nd layer,no.of nodes in 3rd layer,.....,no.of nodes in m layer)<br/>
A_layers-> activation function for all layers except last layer provided options ("relu","sigmoid","identity")<br/>
A_out->activation function for last layer provided options ("relu","sigmoid","identity")<br/>
print_every-> for visualization choose when to print every iteration<br/>
prev_parameters -> is used to save model parameters<br/>
this function returns parameters after training<br/>


if sgd (stochastic gradient descent is selected use this fn)<br/>
use<br/>
 L_layer_model_SGD(X, Y, layers_dims, initialization, A_layers, A_out,prev_parameters, learning_rate, num_iterations, print_cost)<br/>
X->trainig (x_train)<br/>
Y->y label output from onehot function<br/>
layers_dims-> (number of input features,no.of nodes in 1st layer,no.of nodes in 2nd layer,no.of nodes in 3rd layer,.....,no.of nodes in m layer)<br/>
A_layers-> activation function for all layers except last layer provided options ("relu","sigmoid","identity",print_every)<br/>
A_out->activation function for last layer provided options ("relu","sigmoid","identity")<br/>
prev_parameters -> is used to save model parameters<br/>
print_every-> for visualization choose when to print every iteration<br/>
this function returns parameters after training<br/> 


if momentum<br/>
use<br/>
 L_layer_model_GDWithMomentum(X, Y, layers_dims, initialization, A_layers, A_out,prev_parameters, beta, learning_rate, num_iterations, print_cost,print_every)<br/>
X->trainig (x_train)<br/>
Y->y label output from onehot function<br/>
layers_dims-> (number of input features,no.of nodes in 1st layer,no.of nodes in 2nd layer,no.of nodes in 3rd layer,.....,no.of nodes in m layer)<br/>
A_layers-> activation function for all layers except last layer provided options ("relu","sigmoid","identity")<br/>
A_out->activation function for last layer provided options ("relu","sigmoid","identity")<br/>
prev_parameters -> is used to save model parameters<br/>
this function returns parameters after training<br/>
beta ->choose momentum paramete (0-1)<br/>
print_every-> for visualization choose when to print every iteration<br/>
this function returns parameters after training<br/>

if adam<br/>
use<br/>
L_layer_model_Adam(X, Y, layers_dims,initialization, A_layers , A_out ,prev_parameters,beta1 , beta2 ,  epsilon ,learning_rate,num_iterations,  print_cost,print_every)<br/>
X->trainig (x_train)<br/>
Y->y label output from onehot function<br/>
layers_dims-> (number of input features,no.of nodes in 1st layer,no.of nodes in 2nd layer,no.of nodes in 3rd layer,.....,no.of nodes in m layer)<br/>
A_layers-> activation function for all layers except last layer provided options ("relu","sigmoid","identity")<br/>
A_out->activation function for last layer provided options ("relu","sigmoid","identity")<br/>
prev_parameters -> is used to save model parameters<br/> 
this function returns parameters after training <br/>
beta 1 ->choose parameter (0-1)<br/>
beta 2 ->choose parameter (0-1)<br/>
print_every-> for visualization choose when to print every iteration<br/>
this function returns parameters after training <br/>

comment :<br/>
for using minibatch trainig<br/> 
use this function<br/>
L_layer_model_minibatch(X, Y, layers_dims, optimizer_mini_batch, initialization, A_layers, A_out,prev_parameters, mini_batch_size, learning_rate, beta,beta1, beta2, epsilon, num_iterations, print_cost,print_every)<br/>
this function returns parameters after training<br/>

## step 6:<br/>
testing <br/>
use <br/>
L_model_forward(input,parameters,A_layers,A_out)<br/>


## step 7: <br/>
model evaluation<br/>
to draw confusion matrix <br/>
call this function<br/>
confusionmatrix(Y_evalution, Y_evalution_pred)<br/>
this function returns confusion matrix : accuracy , Precision, Recall, F1 score for each class <br/>




## additional features<br/>
## CNN:<br/>
cnn forward propagation is valid in the framework call function <br/>
conv(A_prev, W, b, stride,pad) <br/>
and for pooling call<br/>
pool(A_prev, f,stride, mode = "max")<br/>
to modes available : max and average<br/>
## Data preprocessing<br/>
## LNET5 :<br/>
use lnet function<br/>  
still need preparation<br/>
lnet(x_train)<br/>
x_train-> input image matrix 28x28 variable <br/>







