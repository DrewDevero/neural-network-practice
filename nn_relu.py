import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from packages.datasets.cs23ln import *   #create_data(X, y), show_test_plot()

# show_test_plot()

X, y = create_data(100, 3)
#print(X)
#print(y)

#################################

#U sing rectified linear function as activation function
# all outputs are if x > 0 then output = x else if x <=0 then output = 0
# shows the value of weights flipping sign and biases offsetting 
# solves the vanishing gradient problem of the sigmoid function
# still granular for values of x outputs
# very fast
# fits to non-linear problems (need two or more hidden layers)
# additionally optimizers help determine how neurons are used (as activation or deactivation for eg.)
# eg. for fitting to a sign function: first nueron sets activation and slope, second neuron sets deactivation point
# The area of effect is when both neurons are firing

#################################
np.random.seed(0)

'''
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
    output.append(max(0, i))

print(output)
'''
# want to normalize and scale data to make between -1 < x < 1 to keep data from becomming vary large while retaining its meaning
# want to initialize random small values for weights -1 < w < 1 to start because larger weights will compound as we keep training, causing unruly large numbers/metrics
# biases start at 0 but if this creates a dead network (continually propegating 0 on each train) then we my need to alter the initial bias

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # randn already transposes the weight matrix so we don't need to later for proper training shapes
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # same size as input and normalize
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = LayerDense(2, 5) 
activation1 = ActivationReLU() #currently applying activation to the entire output of layer 1

layer1.forward(X)
#print(layer1.output) # values still negative to positive numbers before processed through the reLU activation function

activation1.forward(layer1.output)
print(activation1.output) # 0 number values returned for all layer1 that is below 0. optimization will tweak to change 0 values. If all go to zero through training, we know the network is dead