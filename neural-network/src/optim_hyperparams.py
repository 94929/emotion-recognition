import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER2013_data

# configs
hidden_dims = [20, 30]
input_dim = 48*48*3
num_classes = 7
dropout = 0
reg = 0.0
seed = 42
weight_scale = 1e-2

# model
model = FullyConnectedNet(
    hidden_dims,
    input_dim,
    num_classes,
    dropout,
    reg,
    weight_scale,
    dtype=np.float32,
    seed=None
)

# dataset
data = get_FER2013_data('/vol/bitbucket/jsh114/emotion-recognition-networks/datasets/FER2013')

# training
best = (0, 0)
initial_starting = 1e-2
minimal_learning = 1e-4
learning_rate = initial_starting
print("entering the loop")
while (learning_rate >= minimal_learning):
    print("Starting with " + str(learning_rate))
    solver = Solver(model,
                    data,
                    update_rule='sgd_momentum',
                    optim_config={
                        'learning_rate': learning_rate,
                        'momentum': 0.0
                    },
                    lr_decay=0.95,
                    batch_size=120,
                    num_epochs=30
                    )
    solver.train()

    learning_rate = learning_rate / 2
    print("accuracy is " + str(solver.val_acc_history))
    best = max(best, (solver.val_acc_history[-1], learning_rate))


print(best)

