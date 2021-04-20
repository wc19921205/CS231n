from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_X = X.shape[0]
    num_class = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_X):
        score = X[i].dot(W)  # 1,C
        score -= np.max(score)
        loss += -score[y[i]] + np.log(np.sum(np.exp(score)))

        s = np.sum(np.exp(score))
        dW[:, y[i]] += -X[i]  # D, C

        for j in range(num_class):
            dW[:, j] += np.exp(score[j]) / s * X[i]

    regularization = 0.5 * reg * np.sum(W * W)
    loss /= num_X
    loss += regularization
    dW = dW / num_X + reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_X = X.shape[0]
    score = X.dot(W)
    correct_class_score = score[np.arange(X.shape[0]), y]
    C = np.max(score, axis=1).reshape(-1, 1)
    loss = np.mean(-correct_class_score + C + np.log(np.sum(np.exp(score - C), axis=1))) + np.sum(W * W) * reg

    co_matrix = np.zeros_like(score)  # N,C
    co_matrix = np.exp(score) / np.exp(score).sum(axis=1).reshape(-1, 1)
    co_matrix[np.arange(X.shape[0]), y] += -1

    dW = X.T.dot(co_matrix) / num_X + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
