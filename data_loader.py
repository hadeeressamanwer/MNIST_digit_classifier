import pickle
import random
import numpy as np
from urllib import request
import matplotlib.pyplot as plt
import gzip
from abc import ABCMeta, abstractmethod

filename = [
	["training_images","train-images-idx3-ubyte.gz"],
	["test_images","t10k-images-idx3-ubyte.gz"],
	["training_labels","train-labels-idx1-ubyte.gz"],
	["test_labels","t10k-labels-idx1-ubyte.gz"]]

def download_mnist():

    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")



def load():
    
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
    
def MakeOneHot(Y, D_out):
    N = Y.shape[0]          
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z

