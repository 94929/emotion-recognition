import numpy as np


def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: A numpy array of shape (N, C)
    - y: A numpy array of shape (N,). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a 
    range of 0 <= y[i] < C

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """
    loss, dlogits = 0, np.zeros_like(logits)
    """
    TODO: Compute the softmax loss and its gradient using no explicit loops
    Store the loss in loss and the gradient in dW. If you are not careful
    here, it is easy to run into numeric instability. Don't forget the
    normalisation!
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    nb_trains, nb_classes = logits.shape
    for i in range(nb_trains):

        # values for each output layer's neuron
        outputs = logits[i]
        
        # for numeric stability, i.e. regularisation
        outputs -= np.max(outputs)
        
        # the expected value for this 
        expected_class = y[i]

        # e_x / e_xs
        exp_output = np.exp(outputs[expected_class])
        sum_exp_outputs = np.sum(np.exp(outputs))

        # add softmaxed values to loss
        loss += -np.log(exp_output / sum_exp_outputs)

        # compute the gradient(i.e. derivative)
        prob_exp_outputs = np.exp(outputs) / sum_exp_outputs
        
        # add current exp values(i.e. costs)
        dlogits[i, :] += prob_exp_outputs

        # reduce cost for the expected class entry
        dlogits[i, expected_class] -= 1

    loss /= nb_trains
    dlogits /= nb_trains
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
