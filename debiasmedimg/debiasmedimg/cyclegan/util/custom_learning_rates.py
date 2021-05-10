import numpy as np
import matplotlib.pyplot as plt


class HalfSteadyHalfLinearDecay:
    def __init__(self, current_epoch=0, max_epochs=200, init_alpha=2e-4):
        """
        Create a learning rate schedule
        :param current_epoch: Current epoch of training when the learning rate is initiated
        :param max_epochs: Maximum epochs the model will be trained
        :param init_alpha: Initial learning rate
        """
        # Store the maximum number of epochs and the base learning rate
        self.maxEpochs = max_epochs
        self.initAlpha = init_alpha
        self.epoch = current_epoch

    def __call__(self):
        """
        Calculate the current learning rate based on the epoch
        :return: Current learning rate according to the schedule
        """
        if self.epoch < (self.maxEpochs / 2):
            self.epoch += 1
            return float(self.initAlpha)
        else:
            # Compute the new learning rate based on polynomial decay
            decay = (1 - ((self.epoch - self.maxEpochs/2)/(self.maxEpochs/2)))
            alpha = self.initAlpha * decay
            self.epoch += 1
            # Return the new learning rate
            return float(alpha)

    def plot(self, n_epochs, title="Learning Rate Schedule"):
        """
        Plot what the learning rate schedule looks like over a given number of epochs
        :param n_epochs: Maximum number of epochs to train
        :param title: Title of the image
        :return: None
        """
        # Compute the set of learning rates for each corresponding
        # Epoch
        epochs = np.arange(0, n_epochs)
        lrs = []
        for i in range(n_epochs):
            lrs.append(self())
        # the learning rate schedule
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")
        plt.show()
