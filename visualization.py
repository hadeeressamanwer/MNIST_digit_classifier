import numpy as np
import matplotlib.pyplot as plt


def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()

    