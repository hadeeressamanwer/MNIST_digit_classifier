import utils_module
import forward_prop
import numpy as np
import matplotlib.pyplot as plt
from init import X_batch

img = X_batch.reshape((28, 28))
plt.imshow(img, cmap="Greys")
plt.show()


# ActivationFunctions = "relu" , "identity" , "sigmoid" , "tanh"
A_layers = "relu"
A_out = "relu"
parameters = utils_module.load('model_parameters.py')
input = np.transpose(X_batch)

Y_pred , cache = forward_prop.L_model_forward(input,parameters,A_layers,A_out)
classification = np.argmax(Y_pred,axis=0)

print("The Number is:" + str(classification))