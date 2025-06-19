########################################################################
# Implements two perceptron classes (using numpy and pytorch, respectively)
# Plotting functions, could be unified but torch needs tensors not numpy
########################################################################


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


######################################################################
# Perceptron class implemented from scratch using numpy
######################################################################


class Perceptron:

    # Initialize the perceptron with learning rate and number of iterations
    def __init__(self, input_dim, learning_rate=0.1, num_iters=10):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.weights = np.random.rand(input_dim) * 0.01
        # self.weights = np.zeros(input_dim)
        # self.bias = 0
        self.bias = np.random.rand() * 0.01
    
    # Activation function
    def activation(self, z):
        return np.where(z >= 0, 1, 0)
    
    def predict(self, X):
        # Z = W.X + b
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)

    def fit(self, X, y):
        # Training the perceptron
        # print(f"Weights: {self.weights}, Bias: {self.bias}")
        for i_step in range(self.num_iters):
            print(f"\nIteration {i_step + 1}/{self.num_iters}")
            # plot_decision_boundary(X, y, self)

            for xi, target in zip(X, y):
                # print(f"Training on input: {xi}, target: {target}")
                z = np.dot(xi, self.weights) + self.bias
                pred = self.activation(z)
                # print(f"Z: {z}, Predicted: {pred}, Target: {target}")
                update = self.learning_rate * (target - pred)
                # print(f"Update: {update}, Update * xi: {update * xi}, Bias Update: {update}")
                self.weights += update * xi
                self.bias += update
                
                print(f"Weights: {self.weights}, Bias: {self.bias}")

def plot_decision_boundary(model, X, y):
        print("Hi!")
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        x1 = np.linspace(x_min, x_max, 200)
        x2 = -(model.weights[0] * x1 + model.bias) / model.weights[1]

        plt.plot(x1, x2, label='Decision Boundary')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Perceptron Decision Boundary")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.grid(True)
        plt.show()


######################################################################
# Perceptron class implemented using PyTorch
######################################################################


class PerceptronTorch(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()

        self.fc = nn.Linear(input_dim, 1) # hard-code single neuron

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

def train_perceptron_torch(model, criterion, optimizer, num_epochs, x_train, y_train):

    for epoch in range(num_epochs):
        
        preds = model(x_train)
        loss = criterion(preds, y_train)

        optimizer.zero_grad()   # zero-out the gradients for the new epoch
        loss.backward()        # calculate the gradients of the loss function
        optimizer.step()        # take a step to optimise
    

def plot_decision_boundary_torch(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                        np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    
    with torch.no_grad():
        Z = model(grid_tensor).reshape(xx.shape).numpy()
    
    plt.contourf(xx, yy, Z >= 0.5, cmap='bwr', alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.xlabel('Sepal length (standardized)')
    plt.ylabel('Petal length (standardized)')
    plt.title('Perceptron Decision Boundary (PyTorch)')
    plt.grid(True)
    plt.show()








