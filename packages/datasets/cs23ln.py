# dataset modified from:
# https://cs23ln.github.io.neural.networks.case.study/
# generates a spiral set of data

import numpy as np

np.random.seed(0)

def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points) #radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    print('cs23ln data set created.')
    return X, y

import matplotlib.pyplot as plt




def show_test_plot():
    X_cs23in, y_cs23in = create_data(100, 3)
    plt.scatter(X_cs23in[:,0], X_cs23in[:,1])
    plt.show()

    plt.scatter(X_cs23in[:,0], X_cs23in[:,1], c=y_cs23in, cmap='brg')
    plt.show()
    