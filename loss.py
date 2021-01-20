import numpy as np
def NLLLoss(Y_pred, Y_true):
    """
    Negative log likelihood loss // softmax
    """
    loss = 0.0
    N = Y_pred.shape[0]
    M = np.sum(Y_pred*Y_true, axis=1)
    for e in M:
        #print(e)
        if e == 0:
            loss += 500
        else:
            loss += -np.log(e)
    return loss/N