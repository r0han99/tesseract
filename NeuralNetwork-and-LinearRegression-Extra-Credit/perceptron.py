import pandas as pd 
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")


class Perceptron:

    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs

    def instantiate_parameters(self, n_features):
        self.weights = np.random.randn(n_features)
        self.bias = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def weighted_sum(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def activate(self, inputs):
        return self.sigmoid(self.weighted_sum(inputs))

    def fit(self, X, y):
        self.instantiate_parameters(X.shape[1])

        # recording loss for visualisation    
        
        loss_records = []
        for epoch in range(self.epochs):
            # Calculate the predicted output
            estimate = self.weighted_sum(X)
            predicted_output = self.activate(X)

            # Calculate the error
            error = y - predicted_output

            # Update the weights and bias
            self.weights += self.lr * np.dot(X.T, error)
            self.bias += self.lr * np.sum(error)

            
            total_error = np.sum((y - predicted_output) ** 2)
            print(f"Epoch {epoch}: Loss = {total_error}")
            loss_records.append(total_error)
        
        return loss_records, self.weights, self.bias

    def predict(self, X):
        return np.round(self.activate(X))

    def accuracy(self, X, y):
        predicted = self.predict(X)
        return np.mean(predicted == y)
        

# Generate data
X, y = make_regression(n_samples=1000, n_features=2, noise=2, random_state=101)

# Converting y to binary
threshold = 0
y_binary = np.where(y >= threshold, 1, 0)

# Normalizing X
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# print(y_binary)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_binary, test_size=0.2, random_state=101)

nn = Perceptron(lr=0.001, epochs=1000)
loss_records, weights, bias = nn.fit(X_train, y_train)
train_acc = nn.accuracy(X_train, y_train)
test_acc = nn.accuracy(X_test, y_test)

print('***'*25)
print(f"Train accuracy: {train_acc}")
print(f"Test accuracy: {test_acc}")


plt.figure(figsize=(7,5))
plt.plot(range(1000), loss_records)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.show()


            



            







       



        







