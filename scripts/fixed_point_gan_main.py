import yaml
import pandas as pd
from sacred import Experiment
from sacred.observers import MongoObserver
from debiasmedimg.fixedpointgan import FixedPointGAN
import debiasmedimg.settings

# Check which server we're working on
with open('configs/config_fixed_point_gan.yaml', 'r') as stream:
    CONFIG = yaml.safe_load(stream)
observer = CONFIG["mongo_observer"]

# Create a sacred experiment
ex = Experiment("FixedPointGAN")

# Connect to the the mongoDB
ex.observers.append(MongoObserver(url=observer))

# Load in the configuration
ex.add_config('configs/config_fixed_point_gan.yaml')
# Standard configuration (e.g. attribute set to x) can be overwritten from the command line


@ex.automain
def main(mongo_observer, dataset_name, transform_between, domain_to_translate, dataset_lowest_folder, load_epoch,
         mode, run_id, image_shape, epochs, base_lr, lambda_adversarial_loss, lambda_cycleloss, lambda_domain_loss,
         lambda_gradient_penalty, lambda_disc_adv, lambda_identityloss, n_batch, n_resnet, additional_losses,
         lambda_additional_losses):

    # Prepare folders
    debiasmedimg.settings.DB_DIR = dataset_lowest_folder
    if load_epoch and not run_id:
        print("Please specify the run id!")
    print("Connected to mongo observer:", mongo_observer)

    # Get the number domain names from the training csv file to ensure that all wanted domains are included
    csv_df = pd.read_csv(dataset_lowest_folder + "train.csv")
    domains = csv_df["origin"].dropna().unique()
    assert all(x in domains for x in transform_between)
    number_of_domains = len(transform_between)

    # Create a Fixed-Point GAN setup
    fixedpointgan = FixedPointGAN(ex, transform_between, load_epoch, run_id, image_shape, number_of_domains, epochs,
                                  base_lr, lambda_adversarial_loss, lambda_cycleloss, lambda_domain_loss,
                                  lambda_gradient_penalty, lambda_disc_adv, lambda_identityloss,
                                  n_batch, n_resnet, additional_losses, lambda_additional_losses)
    if mode == 'train':
        fixedpointgan.train(dataset_lowest_folder + "train.csv", dataset_lowest_folder + "validate.csv",
                            domain_to_translate=domain_to_translate)
    elif mode == 'validate':
        # Evaluate on the validation data
        fixedpointgan.evaluate(dataset_lowest_folder + "validate.csv", val_test=mode,
                               domain_a=domain_to_translate, dataset=dataset_name)
    elif mode == 'test':
        # Evaluate on the test data
        fixedpointgan.evaluate(dataset_lowest_folder + "test.csv", val_test=mode,
                               domain_a=domain_to_translate, dataset=dataset_name)
