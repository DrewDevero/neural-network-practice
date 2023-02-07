import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


#################################

#Using sigmoid function as activation function
# all outputs are between 0 and 1 and are more granular than step
# better for training/more reliable; hlps show how close we are to a certain value for so than step function; so better for calculating loss

#################################
np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

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

# print(0.10 * np.random.randn(4,3))

layer1 = LayerDense(4, 5) 
layer2 = LayerDense(5, 2) #input size must be the previous layer output size

layer1.forward(X)

print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)