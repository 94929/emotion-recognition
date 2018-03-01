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

    logits -= np.max(logits)
    logits_exp = np.exp(logits)
    logits_exp_sum = np.sum(logits_exp, axis=1)
    correct_logits_exp = logits_exp[range(nb_trains), y]

    #print(correct_logits_exp.shape)
    #print(logits_exp.shape)
    #print(logits_exp_sum.shape)
    loss = -np.sum(np.log(correct_logits_exp / logits_exp_sum))
    loss /= nb_trains

    #logits_exp_normalized = logits_exp / logits_exp_sum
    #logits_exp_normalized[range(nb_trains), y] -= 1

    #grad = logits_exp_normalized / nb_trains
    
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits

