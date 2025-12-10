########################################################################
# Uses the numpy MLP implementation
# Can train to binary gate operations
# It is randomly initialised and can get stuck in local minima, giving a 
# rubbish answer (loss ~0.7)
########################################################################

import sys
from pathlib import Path
import numpy as np
import torch
import os

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# change cwd to _ROOT if it's different - copes with running in interactive window
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)

from models.perceptron.MLP import MLP, MLP_torch

if __name__ == "__main__":
    # XOR gate data
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([[0], [1], [1], [0]])  # shape: (4, 1)

    use_torch = True

    if use_torch:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        model = MLP_torch(input_dim=2, hidden_dim=4)
        model.fit(X_tensor, y_tensor)

        print("Hidden layer weights:\n", model.hidden.weight.data)
        print("Hidden layer bias:\n", model.hidden.bias.data)

        print("Output layer weights:\n", model.output.weight.data)
        print("Output layer bias:\n", model.output.bias.data)


        preds = model.predict(X_tensor)

    else:
        model = MLP(input_dim=2, hidden_dim=4, learning_rate=0.2, num_iters=10000)
        model.fit(X, y)

        preds = model.predict(X)


    print("Predictions:", preds.ravel())
    print("Ground truth:", y.ravel())

    print("Hidden layer weights:\n", model.hidden.weight.data)
    print("Hidden layer bias:\n", model.hidden.bias.data)

    print("Output layer weights:\n", model.output.weight.data)
    print("Output layer bias:\n", model.output.bias.data)
