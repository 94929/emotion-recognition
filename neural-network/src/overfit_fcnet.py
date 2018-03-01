import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Overfit the network with 50 samples of CIFAR-10
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
model = FullyConnectedNet([20, 30])
data = get_CIFAR10_data(num_training=50)
solver = Solver(
    model,
    data,
    lr_decay=0.9,
    batch_size=10,
    num_epochs=20,
    num_train_samples=50
)
solver.train()
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
