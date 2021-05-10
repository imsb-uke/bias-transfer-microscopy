import yaml
import pandas as pd
from sacred import Experiment
from sacred.observers import MongoObserver
from debiasmedimg.cyclegan import CycleGAN
import debiasmedimg.settings

# Check which server we're working on
with open('configs/config_cyclegan.yaml', 'r') as stream:
    CONFIG = yaml.safe_load(stream)
observer = CONFIG["mongo_observer"]

# Create a sacred experiment
ex = Experiment("CycleGAN")
# Connect to the the mongoDB
ex.observers.append(MongoObserver(url=observer))

# Load in the configuration
ex.add_config('configs/config_cyclegan.yaml')
# Standard configuration (e.g. attribute set to x) can be overwritten from the command line
# via: python cyclegan_main.py with 'attribute=x'


@ex.automain
def main(mongo_observer, dataset_name, transform_between, dataset_lowest_folder, load_epoch, mode, run_id,
         image_shape, epochs, base_lr, lambda_adversarial_loss, lambda_cycleloss, lambda_identityloss,
         lambda_discriminator_loss, n_batch, n_resnet, additional_losses, lambda_additional_losses):
    # Prepare folders
    debiasmedimg.settings.DB_DIR = dataset_lowest_folder
    if load_epoch and not run_id:
        print("Please specify the run id!")
    print("Connected to mongo observer:", mongo_observer)

    # Get the number domain names from the training csv file to ensure that all wanted domains are included
    csv_df = pd.read_csv(dataset_lowest_folder + "train.csv")
    domains = csv_df["origin"].dropna().unique()
    assert all(x in domains for x in transform_between) and len(transform_between) == 2

    # Create a cycleGAN setup
    cyclegan = CycleGAN(ex, transform_between, load_epoch, run_id, image_shape, epochs, base_lr,
                        lambda_adversarial_loss, lambda_cycleloss, lambda_identityloss, lambda_discriminator_loss,
                        n_batch, n_resnet, additional_losses, lambda_additional_losses)
    if mode == 'train':
        cyclegan.train(dataset_lowest_folder + "train.csv", dataset_lowest_folder + "validate.csv")
    elif mode == 'validate':
        # Evaluate on the validation data
        cyclegan.evaluate(dataset_lowest_folder + "validate.csv", val_test=mode, dataset=dataset_name)
    elif mode == 'test':
        # Evaluate on the test data
        cyclegan.evaluate(dataset_lowest_folder + "test.csv", val_test=mode,
                          dataset=dataset_name, evaluation_direction="AB")
    elif mode == 'transform':
        print(dataset_lowest_folder)
        # Generate transformed images if they don't exist already
        cyclegan.transform_images(dataset_lowest_folder + "test.csv", domain_to_translate=cyclegan.domains[0],
                                  domain_to_translate_to=cyclegan.domains[1], val_test=None)
