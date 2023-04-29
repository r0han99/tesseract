import pandas as pd 
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")


class LinearRegression:


    def __init__(self, lr, epochs):

        self.lr = lr
        self.epochs = epochs
        

    def instantiate_parameters(self):


        self.weights = np.random.randn(self.n_features)
        self.bias = 0


    def predict(self, X):

        self.yhat = np.dot(X, self.weights) + self.bias
        return self.yhat
    
    def compute_cost(self, y_pred, y):

        self.loss = 1/(2*self.samples) * np.sum((y_pred - y)**2)
        return self.loss

    
    def compute_gradients(self, X, y):
        dW = 1/self.samples * np.dot(X.T, (self.yhat - y))
        db = 1/self.samples * np.sum(self.yhat - y)

        return dW, db
    
    def update_params(self, dW, db):

        self.weights = self.weights - (self.lr * dW)
        self.bias = self.bias - (self.lr * db )

        return self.weights, self.bias

    def fit(self, X, y):

        self.samples, self.n_features = X.shape # 1000, 2

        # instantiate initial parameters
        self.instantiate_parameters()

        # Records
        loss_records = []

        # Training Loop
        for epoch in range(self.epochs):

            # Estimate
            y_pred = self.predict(X)
            
            # Compute loss 
            loss = self.compute_cost(y_pred, y)

            # Compute Derivatives
            dw, db = self.compute_gradients(X, y)

            # Update Parameters 
            w, b = self.update_params(dw, db)

            print(f'Epoch: {epoch}, Loss: {loss}')
            loss_records.append(loss)

        print('--'*25)
        print('Training Sequence Completed.')
        print(f'Optimal Weight Vector: [{w}], bias: {b}')

        return loss_records, 


X, y = make_regression(n_samples=1000, n_features=2, noise=2, random_state=101)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

lm = LinearRegression(lr=0.01, epochs=1000)
loss_records = lm.fit(X_train, y_train)


predictions = lm.predict(X_test)
print('Mean Squared Error: ', mse(predictions, y_test))


plt.figure(figsize=(7,5))
plt.plot(range(1000), loss_records[0])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.show()


            



            







       



        







