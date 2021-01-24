import pickle

def save_parameters(filename,parameters):
    open_file = open(filename, "wb")
    pickle.dump(parameters, open_file)
    open_file.close()

def load_parameters(filename):
    open_file = open(filename, "rb")
    loaded_parameters = pickle.load(open_file)
    open_file.close()
    return loaded_parameters


