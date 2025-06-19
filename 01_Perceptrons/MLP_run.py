########################################################################
# Uses the numpy MLP implementation
# Can train to binary gate operations
# It is randomly initialised and can get stuck in local minima, giving a 
# rubbish answer (loss ~0.7)
########################################################################


import numpy as np
import MLP as mlp

# OR gate data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])  # shape: (4, 1)

# Train MLP
model = mlp.MLP(input_dim=2, hidden_dim=4, learning_rate=0.1, num_iters=10000)
model.fit(X, y)

# Predict
preds = model.predict(X)
print("Predictions:", preds.ravel())
print("Ground truth:", y.ravel())
