from data_loader import *
import random


def init():
    download_mnist()
    save_mnist()
def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N-batch_size)
    return X[i:i+batch_size], Y[i:i+batch_size]

init()
X_train, Y_train, X_test, Y_test = load()
Y_evalution = Y_test



batch_size = 1
D_out=10

X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
Y_train = MakeOneHot(Y_train, D_out)
Y_batch = MakeOneHot(Y_batch, D_out)
Y_test  = MakeOneHot(Y_test, D_out)


