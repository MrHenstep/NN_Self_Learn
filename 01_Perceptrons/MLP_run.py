########################################################################
# Uses the numpy MLP implementation
# Can train to binary gate operations
# It is randomly initialised and can get stuck in local minima, giving a 
# rubbish answer (loss ~0.7)
########################################################################


import numpy as np
import MLP as mlp
import torch

# XOR gate data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])  # shape: (4, 1)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


# Train MLP
model = mlp.MLP(input_dim=2, hidden_dim=4, learning_rate=0.2, num_iters=10000)
# model = mlp.MLP_torch(input_dim=2, hidden_dim=4)

model.fit(X, y)
# model.fit(X_tensor, y_tensor)

# Predict
preds = model.predict(X)
# preds = model.predict(X_tensor)


print("Predictions:", preds.ravel())
print("Ground truth:", y.ravel())

