from datasets.mnist import MNIST

from nn.model import Sequential
from nn.activations import relu, tanh, sigmoid
from nn.layers import Dense
from nn.loss import MeanSquaredError
from nn.optimizer import GradientDescent

from nn.preprocessing import categorical_encoding, transform_input_data

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# loading MNIST
data = MNIST(mode='full')
train_data, test_data = data.dataset
X_train, y_train = train_data
X_test, y_test = test_data

print('Train:', X_train.shape, y_train.shape)
print('Test:', X_test.shape, y_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test_ = scaler.transform(X_test)

X_test_ = transform_input_data(X_test_)
X_train_ = transform_input_data(X_train)

y_train_ = categorical_encoding(y_train)
y_train_ = transform_input_data(y_train_)


nn = Sequential(input_shape=(1, 784),
                layers=[
                    Dense(128, activation=relu),
                    Dense(10, activation=tanh)],
                optimizer=GradientDescent(0.1),
                loss=MeanSquaredError())

nn.info()

nn.fit(X_train_, y_train_, epochs=2)


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
