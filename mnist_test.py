from datasets.mnist import MNIST, normalize_mnist

from nn.dataloader import DataLoader
from nn.model import Sequential
from nn.activations import leaky_relu, relu, softmax, tanh, sigmoid
from nn.layers import Dense
from nn.loss import MeanSquaredError, CategoricalCrossEntropy
from nn.optimizer import GradientDescent

from nn.preprocessing import categorical_encoding, transform_input_data

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# loading MNIST
data = MNIST(mode='full')
train_data, test_data = data.dataset
X_train, y_train = train_data
X_test, y_test = test_data

print('Train:', X_train.shape, y_train.shape)
print('Test:', X_test.shape, y_test.shape)

X_train = normalize_mnist(X_train)
X_test_ = normalize_mnist(X_test)

X_test_ = transform_input_data(X_test_)
X_train_ = transform_input_data(X_train)

y_train_ = categorical_encoding(y_train)
y_train_ = transform_input_data(y_train_)



print('Training shapes')
print('X_train:', X_train_.shape)
print('y_train:', y_train_.shape)

# batch_size = 1
# dl_train = DataLoader(X_train_, y_train_)
# dl_test = DataLoader(X_test_, y_test)
# dl_train.batch(batch_size)
# dl_test.batch(batch_size)


nn = Sequential(input_shape=(1, 784),
                layers=[
                    Dense(200, activation=relu),
                    Dense(10, activation=sigmoid)],
                optimizer=GradientDescent(0.15),
                loss=MeanSquaredError())

nn.info()

nn.fit(X_train_, y_train_, epochs=10)


y_hats = []
for x_i in X_test_:
    y_hat = nn.predict(x_i)
    y_hats.append(np.argmax(y_hat))

acc = accuracy_score(y_test, y_hats)
print('Accuracy:', acc)

cm = confusion_matrix(y_test, y_hats)
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True)
plt.show()
