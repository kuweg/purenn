from sklearn.datasets import make_classification, make_blobs
import matplotlib.pyplot as plt
import numpy as np

from nn.model import Sequential
from nn.layers import InputLayer, Dense, WeightsLayer
from nn.activations import relu, tanh, apply_function_to_nparray
from nn.losses import BinaryCrossEntropy, mean_squared_error

samples = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, flip_y=-1)

n_split = int(np.round(len(samples[0]) * 0.7))
x = samples[0]
y = samples[1].reshape(-1, 1)
x_train, x_test = x[:n_split], x[n_split:]
y_train, y_test = y[:n_split], y[n_split:]
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

nn = Sequential(InputLayer(2),
                Dense(3, activation = tanh),
                Dense(1, activation = tanh))

alpha = 0.01
wl = list(nn.weights.__dict__.values())[:-1]
for epoch in range(100):
    for x_i, y_i in zip(x_train, y_train):
        y_hat = nn.forward(x_i)
        error = mean_squared_error(y_i, y_hat)

        for weight_layer in reversed(wl):
            # print('<{}>'.format(weight_layer))
            error = weight_layer.backprob(error)
        
    print("Epoch {}: Error - {}".format(epoch, sum(error)))
    
    
y_hat = np.array([])
for x_i in x_test:
    y_hat = np.append(y_hat, nn.forward(x_i))
  


y_hat = apply_function_to_nparray(y_hat, lambda value: 1 if value >= 0.5 else 0)  
print(y_test.reshape(1, -1))
print(y_hat)

from sklearn import metrics

cm = metrics.confusion_matrix(y_test, y_hat)
print(cm)