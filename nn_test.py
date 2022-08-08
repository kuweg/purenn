from tqdm import tqdm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from nn.model import Sequential
from nn.layers import InputLayer, Dense, WeightsLayer
from nn.activations import relu, tanh, apply_function_to_nparray, sigmoid
from nn.losses import BinaryCrossEntropy, mean_squared_error

# samples = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, flip_y=-1)

# n_split = int(np.round(len(samples[0]) * 0.7))
# x = samples[0]
# y = samples[1].reshape(-1, 1)
# x_train, x_test = x[:n_split], x[n_split:]
# y_train, y_test = y[:n_split], y[n_split:]
# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


data = load_iris()
x = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=4)

print('Train shapes: {} | {}'.format(X_train.shape, y_train.shape))
print('Test shapes: {} | {}'.format(X_test.shape, y_test.shape))

print('Data example:', '\n', '{} -> {}'.format(X_train[0], y_train[0]))

nn = Sequential(InputLayer(4),
                Dense(3, activation = tanh),
                Dense(1, activation = tanh))



alpha = 0.01
errors = []
wl = list(nn.weights.__dict__.values())[:-1]
n_epochs = 10
for epoch in range(n_epochs):
    errors_buffer = []
    for x_i, y_i in zip(X_train ,y_train):
        y_hat = nn.forward(x_i)
        
        print("<{} | {}> -> {}".format(x_i, y_i, y_hat))  
        loss = y_hat - y_i
        
        errors_buffer.append(loss)

        error = mse_prime(y_i, y_hat)
        for weight_layer in reversed(wl):
            # print('<{}>'.format(weight_layer))
            error = weight_layer.backward(error)
    errors.append(errors_buffer)
    print("Epoch {}: Error - {}".format(epoch, np.mean(errors_buffer)))
    
errors = np.squeeze(np.array(errors))
    
    
y_hat = np.array([])
for x_i in X_test:
    y_hat = np.append(y_hat, nn.forward(x_i))
  


y_hat = apply_function_to_nparray(y_hat, lambda value: 1 if value >= 0.5 else 0)  
print(y_test.reshape(1, -1))
print(y_hat)

from sklearn import metrics

cm = metrics.confusion_matrix(y_test, y_hat)
print(cm)