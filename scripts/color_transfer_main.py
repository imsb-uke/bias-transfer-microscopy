import os
import pandas as pd
import numpy as np
from debiasmedimg.color_transfer import ColorTransfer
import debiasmedimg.settings
from sacred import Experiment

# Create a sacred experiment
ex = Experiment("Color Transfer")

# Load in the configuration
ex.add_config('configs/config_color_transfer.yaml')

@ex.automain
def main(dataset_name, domain_a, domain_b, dataset_lowest_folder, mode, seed):
    debiasmedimg.settings.DB_DIR = dataset_lowest_folder

    if mode == "validate":
        # Apply color transfer, save images and evaluate the metrics
        A_validate = ColorTransfer(dataset_lowest_folder + "validate.csv", seed)
        A_validate.apply_transfer(domain_a, domain_b)
        A_validate.evaluate(True, dataset_lowest_folder + "validate.csv", domain_a, domain_b,
                            dataset=dataset_name)
    elif mode == "test":
        # Apply color transfer, save images and evaluate the metrics
        A_test = ColorTransfer(dataset_lowest_folder + "test.csv", seed)
        A_test.apply_transfer(domain_a, domain_b)
        A_test.evaluate(False, dataset_lowest_folder + "test.csv", domain_a, domain_b,
                        dataset=dataset_name)
