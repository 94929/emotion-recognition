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
# define model and data
model = FullyConnectedNet(hidden_dims=[20, 30])
data = get_CIFAR10_data(num_training=50)

# define solver which helps us to train our model using the data
solver = Solver(
    model,
    data,
    num_epochs=20,
    num_train_samples=50
)

# train our model using the solver
solver.train()
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

