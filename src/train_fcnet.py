import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50% 
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
#define model and data
model = FullyConnectedNet(hidden_dims=[20, 30])
data = get_CIFAR10_data()

# define solver which helps us to train our model using the data
solver = Solver(
    model,
    data,
    lr_decay=0.95,    
    num_epochs=30,
    batch_size=120
)

# train the model
solver.train()
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
