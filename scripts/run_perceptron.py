########################################################################
# Demonstrates the two perceptron implementations (numpy, pytorch)
# by training on iris data and plotting. 
########################################################################

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.perceptron import perceptron as ptron

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim


if __name__ == "__main__":

    # Load dataset
    iris = datasets.load_iris()
    X = iris.data[:100, [0, 2]]  # use 2 features: sepal length and petal length
    y = iris.target[:100]        # only setosa and versicolor

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ####################################################################### Create perceptron (numpy version), train, and plot
    ######################################################################

    model = ptron.Perceptron(input_dim=X_train.shape[1], learning_rate=0.01, num_iters=100)
    model.fit(X_train, y_train)

    ptron.plot_decision_boundary(model, X_train, y_train)

    ####################################################################### Make tensor versions of the data sets for torch
    ######################################################################

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    ####################################################################### Create perceptron (torch version), train, and plot
    ######################################################################

    model = ptron.PerceptronTorch(input_dim=X_train.shape[1])

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    ptron.train_perceptron_torch(model, criterion=criterion, optimizer=optimizer, num_epochs=500, x_train=X_train_tensor, y_train=y_train_tensor)

    ptron.plot_decision_boundary_torch(model, X_train, y_train)
