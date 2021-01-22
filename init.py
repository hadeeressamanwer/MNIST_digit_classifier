from data_loader import *


def init():
    download_mnist()
    save_mnist()
def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N-batch_size)
    return X[i:i+batch_size], Y[i:i+batch_size]
#init()
X_train, Y_train, X_test, Y_test = load()
#X_train, X_test = X_train/float(255), X_test/float(255)
#X_train -= np.mean(X_train)
#X_test -= np.mean(X_test)
batch_size = 2000
D_out=10

X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
Y_train = MakeOneHot(Y_train, D_out)
Y_batch = MakeOneHot(Y_batch, D_out)
#D_in = 784
#print (Y_train)
#print (Y_batch)
#print (X_train.shape)
#print(Y_train.shape)
