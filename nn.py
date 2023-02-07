import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#print('python {}'.format(sys.version))
#print('numpy {}'.format(np.__version__))
#print('matplotlib {}'.format(matplotlib.__version__))

# creating a neuron (as a connection to every previous neuron)
# inputs are the outputs from previous layers
# each neuron has a unique weight and bias 
# every neuron has one bias even when taking multiple inputs with multiple weights from previous neurons
# arbitrary neuron below with arbitrary weights and biases

# simulating inputs from a first input layer into one first hidden layer neuron
inputs = [1.3, 5.3, 2.8]

weights1 = [3.4, 2.3, 8.5]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.25, -0.45, 0.17, 0.89]

bias1 = 2
bias2 = 3
bias3 = 0.5

#output = sum of inputs times weights, then add the bias

def generate_output(a,b,c):
    iter = len(a)
    out = 0
    for i in range(0, iter):
        out += a[i] * b[i]
    out += c
    return out

output = generate_output(inputs, weights1, bias1)

print(output)

def add_neuron(i, a, w, b):
    i.append(a)
    w.append(b)
    print(i, w)

# testing adding an initial neuron with input and weight
add_neuron(inputs, 4.3, weights1, 3.7)

# looking at the output layer of a four layer simple neural network
# Let's say 3 input neurons, 4 first hidden layer neurons, 4 second hidden layer neurons, and 3 output layer neurons
# Looking at second hidden layer to output layer
# This means four unique inputs going into each output neuron, each output neuron has its own unique weight set (weight per input) and bias (bias per output neuron)
# So one input put to 3 differenct weight sets and 3 different biases
"""
inputs = [1.3, 5.3, 2.8] # from hidden layer two four neurons

weights1 = [3.4, 2.3, 8.5] # for output neruon 1
weights2 = [0.5, -0.91, 0.26, -0.5] # for output neruon 2
weights3 = [-0.25, -0.45, 0.17, 0.89] # for output neruon 3

bias1 = 2 # for output neruon 1
bias2 = 3 # for output neruon 2
bias3 = 0.5 # for output neruon 3
"""

output = [generate_output(inputs, weights1, bias1), generate_output(inputs, weights2, bias2), generate_output(inputs, weights3, bias3)]
print(output)

# see that the inputs are coming from previous neurons so they don't manually change, the change comes from our manipulating weights and biases with back propogation

# zip commbines lists element-wise, using below for iteration

# inputs are the same
# now using 2d list for weights and 1d list iases

weights = [weights1,
            weights2,
            weights3]

biases = [bias1, bias2, bias3]

layer_outputs = []
for n_weights, n_biases in zip(weights, biases):
    layer_outputs.append(generate_output(inputs, n_weights, n_biases))

print(layer_outputs)

# weights change the magnitude 

# bias offsets the value just like constant b in a liear equation

# using numpy 

outputs = np.dot(weights, inputs) + biases

# dot product looks at the shape of the first argument, in this case (2, 4) of weights and iterates over each vector while applying the multiplication of the second argument elements then applying the remaining (in this case bias) for each completed iteration before moving onto the next vector 

print('Using Dot Product: {}'.format(outputs))

# features from a single sample vs batches

inputs = [inputs,
            [2.0, -3.4, 4.5, -2.3],
            [-3.2, -1.4, -5.7, 6.3]]

# need to transpose to match shapes of two arguments where num of elements in vector arg 1 = shape of arg 2
# can go with inputs or weights now since they have the same shape
# dot changes lists to np arrays so we need to convert list to np array, then transpose within dot
# transpose works for dot so that each element of each row matches with each element of the nth index of the second argument vectors

outputs = np.dot(inputs, np.array(weights).T) + biases

print(outputs)

# adding another layer 

weights2 = [[3,4,5,],
            [0.4, 0.3, -0.6],
            [0.3, 5, 7.6]]

biases2 = [0.4, 5, -0.8]

layer1_outputs = outputs
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)