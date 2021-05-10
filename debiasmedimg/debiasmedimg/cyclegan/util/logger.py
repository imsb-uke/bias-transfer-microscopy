import numpy as np


class Logger:
    def __init__(self, ex, names, mode):
        """
        Initiate a logger given the experiment, the losses/metrics to log and the training mode
        :param ex: Experiment to log to
        :param names: List of strings of loss/metric names
        :param mode: Training/validating/testing right now
        """
        self.ex = ex
        self.mode = mode
        self.names = names.copy()
        if mode == 'validate':
            for ix, name in enumerate(self.names):
                first_part, last_part = name.split('_loss')
                new_name = first_part + '_val_loss'
                self.names[ix] = new_name
        elif mode == 'evaluate_val':
            for ix, name in enumerate(self.names):
                new_name = name + '_val'
                self.names[ix] = new_name
        elif mode == 'evaluate_test':
            for ix, name in enumerate(self.names):
                new_name = name + '_test'
                self.names[ix] = new_name
        # Initiate empty batch
        self.batch_values = []
        for _ in names:
            self.batch_values.append([])

    def log_batch(self, batch):
        """
        Add losses/metrics of one batch to the log
        :batch: Losses/metrics to save
        :return: None
        """
        for ix, value in enumerate(batch):
            self.batch_values[ix].append(value)

    def log_specific_batch(self, batch, ids):
        """
        Add specific losses/metrics of one batch to the log
        :param batch: Losses/metrics to save
        :param ids: List of ids of the losses/metrics to save
        :return: None
        """
        for counter, idx in enumerate(ids):
            self.batch_values[idx].append(batch[counter])

    def get_batch_mean(self):
        """
        Receive the mean batch losses/metrics
        :return: Mean losses/metrics as a list
        """
        mean_values = []
        for ix, values in enumerate(self.batch_values):
            mean_values.append(np.mean(values))
        return mean_values

    def reset_batch(self):
        """
        Clear the losses/metrics of the batches
        :return: None
        """
        for ix, values in enumerate(self.batch_values):
            self.batch_values[ix] = []

    def log_to_ex(self, epoch, learning_rate=None):
        """
        Log the losses/metrics to sacred
        :param epoch: Current epoch number
        :param learning_rate: During training we also want to log the learning rate
        :return: None
        """
        mean_batch = self.get_batch_mean()
        # Add full summary to sacred
        for ix, mean in enumerate(mean_batch):
            self.ex.log_scalar(self.names[ix], mean, step=epoch)
        if self.mode == 'train':
            self.ex.log_scalar("learning_rate", learning_rate, step=epoch)
