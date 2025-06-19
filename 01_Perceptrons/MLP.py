########################################################################
# Implements a two-layer MLP (one hidden, one output)
# using only numpy, so doing the gradients by hand
########################################################################


import numpy as np

class MLP:

    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 learning_rate=0.1, 
                 num_iters=10000):
        
        self.lr = learning_rate
        self.num_iters = num_iters

        # initialise weights
        self.W_hidden = np.random.rand(input_dim, hidden_dim) * 0.1
        self.b_hidden = np.zeros((1, hidden_dim))
        self.W_output = np.random.rand(hidden_dim, 1) * 0.1
        self.b_output = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s*(1-s)
    

    # tanh itself is implemented in numpy

    def tanh_deriv(self, x):
        return 1 - np.tanh(x)**2
    
    def forward(self, X):
        self.z1 = X @ self.W_hidden + self.b_hidden
        self.h = np.tanh(self.z1)
        self.z2 = self.h @ self.W_output + self.b_output
        self.y_hat = self.sigmoid(self.z2)
        return self.y_hat
    
    # BCE loss
    def compute_loss(self, y_true, y_pred):
        eps = 1e-8
        # print(y_true, y_pred)
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))   

    # backward does the gradient calculation AND then updates the parameters
    def backward(self, X, y):
        m = y.shape[0]
        
        dL_dz2 = self.y_hat - y
        dL_dW_output = self.h.T @ dL_dz2 / m                    # (hidden,1)
        dL_db_output = np.sum(dL_dz2, axis=0, keepdims=True) / m

        dL_dh = dL_dz2 @ self.W_output.T                        # (m, hidden)
        dL_dz1 = dL_dh * self.tanh_deriv(self.z1)         # (m, hidden)
        dL_dW1 = X.T @ dL_dz1 / m                         # (input, hidden)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True) / m

        # now update the parameters using these gradients
        self.W_output -= self.lr * dL_dW_output
        self.b_output -= self.lr * dL_db_output
        self.W_hidden -= self.lr * dL_dW1
        self.b_hidden -= self.lr * dL_db1
    
    def fit(self, X, y, verbose=True):

        for i in range(self.num_iters):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y)

            # print(f"Epoch {i}, Loss: {loss:.4f}")

            if verbose and i % (self.num_iters // 10) == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")
    
    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred >= 0.5).astype(int)
    

####################################################################################
# Can also implement with Pytorch: but you need a training routine, which I haven't
# done
# Note how simple it is - use linear layers 
# and then implement the activations in the forward method. 
####################################################################################

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=4):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))       # same as NumPy version
        x = torch.sigmoid(self.output(x))    # sigmoid for binary classification
        return x
