import os
from os import listdir
from copy import deepcopy
import random
import time
import datetime
import numpy as np
import matplotlib
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import tensorflow as tf
import debiasmedimg.settings as settings
from .util import define_generator, define_discriminator, define_cyclegan_discriminator, \
    cut_out_files
from debiasmedimg.cyclegan.util import get_filenames, normalize_for_display, normalize_for_evaluation, ssim_score, \
    get_fid, laplacian_upsampling, get_sample_from_path, get_real_samples, Logger, save_to_csv, get_all_samples, \
    create_patches


class FixedPointGAN:
    """
    A class that encapsulates a starGAN implementation
    The losses used for training depend on  given parameters during the initialization
    """

    def __init__(self, ex, domain_names, load_epoch, run_id, image_shape, n_labels, epochs, base_lr,
                 lambda_adversarial_loss, lambda_cycleloss, lambda_domain_loss, lambda_gradient_penalty,
                 lambda_disc_adv, lambda_identityloss, n_batch, n_resnet, additional_losses, lambda_additional_losses):
        """
        Create a Fixed-Point GAN
        :param ex: Sacred experiment to log to
        :param domain_names: Names of the domains the network is transforming between
        :param load_epoch: Which episode to load, if run id already exists
        :param run_id: ID of the model if a model is loaded else None
        :param image_shape: Shape of the input image
        :param n_labels: How many classes to transfer between
        :param epochs: Number of epochs to train
        :param base_lr: Learning rate to start training with
        :param lambda_adversarial_loss: Lambda of the adversarial loss
        :param lambda_cycleloss: Lambda of the cycle loss
        :param lambda_domain_loss: Lambda of the domain loss
        :param lambda_gradient_penalty: Lambda of the gradient penalty
        :param lambda_identityloss: Lambda of the conditional identity loss
        :param n_batch: Number of training samples per batch
        :param n_resnet: Number of resNet blocks in the generator
        :param additional_losses: Any losses added to the system
        :param lambda_additional_losses: Lambdas of the additional losses
        """
        # Save parameters
        assert len(domain_names) == n_labels
        self.ex = ex
        self.load_epoch = load_epoch
        self.run_id = run_id
        self.image_shape = image_shape
        self.n_labels = n_labels
        self.domains = domain_names
        self.labels = self.domains_to_labels(self.domains)
        self.epochs = epochs
        self.base_lr = base_lr
        self.lambda_adversarial_loss = lambda_adversarial_loss
        self.lambda_cycleloss = lambda_cycleloss
        self.lambda_domain_loss = lambda_domain_loss
        self.lambda_gradient_penalty = lambda_gradient_penalty
        self.n_batch = n_batch
        self.n_resnet = n_resnet
        self.additional_losses = additional_losses
        self.lambda_additional_losses = lambda_additional_losses
        self.lambda_disc_adv = lambda_disc_adv
        self.lambda_identityloss = lambda_identityloss

        # Create run-id
        if not self.run_id:
            self.run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.run_id = self.run_id

        # Initialize learning rate
        self.curr_lr = self.base_lr

        # Create models
        self.generator = define_generator(self.image_shape, [self.n_labels], target_units=self.image_shape[0],
                                          out_channels=self.image_shape[2], n_resnet=self.n_resnet)

        # Image -> real/fake (patch), class label
        # self.discriminator = define_discriminator(self.image_shape, self.n_labels)
        self.discriminator = define_cyclegan_discriminator(self.image_shape, self.n_labels)
        # Create one optimizer per model
        self.generator_optimizer = tf.keras.optimizers.Adam(self.base_lr, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.base_lr, beta_1=0.5)

        # Create checkpoint manager for saving the models during training
        self.checkpoint_path = settings.OUTPUT_DIR + "/checkpoints/" + self.run_id + "/train"
        # Define what to store in the checkpoint
        self.ckpt = tf.train.Checkpoint(generator=self.generator,
                                        discriminator=self.discriminator)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=10)
        # If a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            if load_epoch:
                ckpt_to_restore = [s for s in self.ckpt_manager.checkpoints if "ckpt-" + str(self.load_epoch) in s][0]
            else:
                ckpt_to_restore = self.ckpt_manager.latest_checkpoint
            print(ckpt_to_restore)
            status = self.ckpt.restore(ckpt_to_restore)
            print('Latest checkpoint restored!!')
            # The number of the checkpoint indicates how many epochs have been trained so far
            path, id_and_checkpoint = ckpt_to_restore.split('checkpoints/')
            self.start_epoch = int(id_and_checkpoint.split('-')[2])
            status.assert_existing_objects_matched()
        else:
            self.start_epoch = 0

        # Initiate batch losses
        losses = ["discriminator_loss", "discriminator_adversarial_loss", "discriminator_gradient_penalty_loss",
                  "domain_class_real_loss", "generator_loss", "generator_adversarial_loss",
                  "domain_class_fake_loss", "conditional_identity_loss", "reconstruction_loss"]
        evaluation_metrics = ["ssim_inout_a", "fid_orig", "fid_b"]
        self.training_logger = Logger(ex, losses, mode='train')
        self.validation_logger = Logger(ex, losses, mode='validate')
        self.eval_val_logger = Logger(ex, evaluation_metrics, mode='evaluate_val')
        self.eval_test_logger = Logger(ex, evaluation_metrics, mode='evaluate_test')
        # Initiate for visualization of training
        self.vis_img_in = None
        # Initiate for identifying best training epoch
        self.best_ckpt_loss = float('Inf')
        self.init_val_loss_negative = None

    def update_lr(self, epoch):
        """
        Update the learning rate depending on the epoch
        :param epoch: Current epoch
        :return: None
        """
        # Dealing with the learning rate as per the epoch number
        if epoch < self.epochs / 2:
            self.curr_lr = self.base_lr
        else:
            decay = (1 - ((epoch - self.epochs / 2) / (self.epochs / 2)))
            self.curr_lr = self.base_lr * decay
        # Set the learning rates of the optimizers
        self.generator_optimizer.lr.assign(self.curr_lr)
        self.discriminator_optimizer.lr.assign(self.curr_lr)

    def get_random_domains(self, batch_size):
        """
        Return an array of random domain names in the size of the batch
        :param batch_size: Size of one batch
        :return: Array of domain names
        """
        ix = np.random.randint(0, self.n_labels, batch_size)
        return np.array(self.domains)[ix]

    def get_random_domains_without_original(self, batch_size, original_domains):
        """
        Return an array of random domain names in the size of the batch
        :param batch_size: Size of one batch
        :param original_domains: Domains to translate from
        :return: Array of domain names
        """
        domains_b = []
        for i in range(batch_size):
            ix = np.random.randint(0, self.n_labels, 1)
            # Don't convert to the original domain
            while np.array(self.domains)[ix] == original_domains[i]:
                ix = np.random.randint(0, self.n_labels, 1)
            domains_b.append(np.array(self.domains)[ix])
        return np.array(domains_b)

    def labels_to_domains(self, labels):
        """
        Get the domain name given the one-hot encoded label
        :param labels: one-hot encoded label
        :return: domain_name
        """
        domains = []
        for label in labels:
            n_label = np.where(label == 1)[0]
            domain_name = self.domains[n_label]
            domains.append(domain_name)
        return np.array(domains)

    def domains_to_labels(self, domain_names):
        """
        Get the one-hot encoded label given the domain name
        :param domain_names: name of the domain
        :return: label of the domain
        """
        labels = []
        for domain in domain_names:
            n_label = self.domains.index(domain)
            label = np.zeros(self.n_labels)
            label[n_label] = 1
            label = np.array(tf.cast(label, tf.float32))
            labels.append(label)
        return np.array(labels)

    def visualize_performance(self, train_files, domain_in, epoch):
        """
        Put out a plot showing how the generator transforms the images after each epoch
        :param train_files: Paths to training images
        :param domain_in: Name of the domain to translate
        :param epoch: Current epoch of the training process
        :return:
        """
        if epoch == 'init':
            # Find the first entry that contains the name of the domain which we want to translate
            domain_id = self.domains.index(domain_in)
            img_file_in = train_files[domain_id][0]
            _, self.vis_img_in, _ = get_sample_from_path(img_file_in)
        generated_images = []
        titles = []
        # Get transformed images from generators
        for i in range(self.n_labels):
            label = np.array([self.labels[i]])
            gen_image = self.generator([self.vis_img_in, label], training=False)
            generated_images.append(gen_image)
            titles.append(self.domains[i])
        # Normalize images to [0,1]
        img_in = normalize_for_display(self.vis_img_in)
        for ix, image in enumerate(generated_images):
            generated_images[ix] = normalize_for_display(image)
        # Create figure for displaying images
        _ = plt.figure(figsize=(20, 8 * len(generated_images)))
        _ = GridSpec(self.n_labels, 2)
        # Create "from" column subplot
        plt.subplot2grid((self.n_labels, 2), (0, 0), colspan=1, rowspan=1)
        plt.title(domain_in)
        plt.imshow(img_in[0])
        # Create "to" column subplots
        for i in range(len(generated_images)):
            plt.subplot2grid((self.n_labels, 2), (i, 1), colspan=1, rowspan=1)
            plt.title(titles[i])
            plt.imshow(generated_images[i][0])
        path = settings.OUTPUT_DIR + "/plots/" + self.run_id + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + str(epoch) + ".png")
        plt.close()

    def discriminator_adv_loss(self, prob_real_is_real, prob_fake_is_real):
        """
        Loss of the discriminator
        :param prob_real_is_real: Loss of the discriminator on real samples
        :param prob_fake_is_real: Loss of the discriminator on fake samples
        :return: total loss of the discriminator divided by two
        """
        # p(real/fake) = 1 if it is real, 0 if it is fake
        if self.lambda_gradient_penalty == 0:
            # GAN objective
            loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            real_is_real_loss = loss_obj(tf.ones_like(prob_real_is_real), prob_real_is_real)
            fake_is_real_loss = loss_obj(tf.zeros_like(prob_fake_is_real), prob_fake_is_real)
        else:
            # WGAN objective
            real_is_real_loss = -tf.reduce_mean(prob_real_is_real)
            fake_is_real_loss = tf.reduce_mean(prob_fake_is_real)
        loss = real_is_real_loss + fake_is_real_loss
        return self.lambda_disc_adv * loss

    def calc_cycle_loss(self, real_image, cycled_image):
        """
        Calculate the loss between the real input and the cycled output (ideally identical)
        :param real_image: Real input
        :param cycled_image: Cycled output generated from the real input
        :return: cycle loss of the network
        """
        loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.lambda_cycleloss * loss

    def generator_adv_loss(self, prob_fake_is_real):
        """
        Calculate the generator loss (whether the discriminator was able to tell fake from real images)
        :param prob_fake_is_real: Probabilities predicted by the discriminator
        :return: generator loss
        """
        if self.lambda_gradient_penalty == 0:
            # GAN objective
            loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            loss = loss_obj(tf.ones_like(prob_fake_is_real), prob_fake_is_real)
        else:
            loss = -tf.reduce_mean(prob_fake_is_real)
        return self.lambda_adversarial_loss * loss

    def domain_class_loss(self, real_label, predicted_label):
        """
        Calculate the domain classification loss between the real labels and predicted labels
        :param real_label: Real labels of images
        :param predicted_label: Predicted labels
        :return: Domain classification loss
        """
        # Softmax for one-hot encoded vectors (otherwise sigmoid)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=real_label, logits=predicted_label)
        loss = tf.reduce_mean(loss)
        return self.lambda_domain_loss * loss

    def wgan_gp_loss(self, real_img, fake_img):
        """
        Calculate the gradient penalty from Wasserstein-GANs to stabilize training
        :param real_img: Real images
        :param fake_img: Generated images
        :return: Gradient penalty
        """
        alpha = tf.random.uniform(shape=[self.n_batch, 1, 1, 1], minval=0., maxval=1.)
        interpolates = alpha * fake_img + (1. - alpha) * real_img
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(interpolates)
            disc_interpolates = self.discriminator(interpolates)[0]
        # https://github.com/WangZesen/WGAN-GP-Tensorflow-v2/blob/master/train.py :
        gp_gradients = inner_tape.gradient(disc_interpolates, interpolates)
        gp_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)
        return self.lambda_gradient_penalty * gradient_penalty

    def conditional_identity_loss(self, image_in, image_out):
        """
        Calculates the conditional identity loss (is only called for same-same transformation)
        :param image_in: Image that was translated
        :param image_out: Translated image
        :return: Identity loss
        """
        loss = tf.reduce_mean(tf.abs(image_in - image_out))
        return self.lambda_identityloss * loss

    def additional_identity_loss(self, real_image, same_image, epoch, final_epoch):
        """
        Additional identity loss as added by de Bel et al.
        :param real_image: Image put into generator
        :param same_image: Image produced by generator
        :param epoch: Current epoch
        :param final_epoch: Final epoch where the additional identity loss is used
        :return: Loss
        """
        lambda_id = self.additional_losses.index("add_identity_loss")
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        # Loss is reduced to zero over the first epochs
        lambda_add_identity_loss = self.lambda_additional_losses[lambda_id] - \
            epoch * self.lambda_additional_losses[lambda_id] / final_epoch
        return lambda_add_identity_loss * loss

    def ms_ssim_loss(self, image_a, cycled_image_a):
        """
        MS-SSIM loss as used by Armanious et al. for MR images
        :param image_a: Image of domain A
        :param cycled_image_a: Cycled image of domain A
        :return: MS-SSIM loss of one domain
        """
        # max-val = difference between the maximum the and minimum allowed values,
        # images here are normalized to be in range [-1,1] -> max_val = 2
        ms_ssim = (1 - tf.image.ssim_multiscale(image_a, cycled_image_a, max_val=2))
        lambda_id = self.additional_losses.index("ms_ssim_loss")
        return self.lambda_additional_losses[lambda_id] * ms_ssim

    def ma_structure_loss(self, images, generated_images, patch_size=16, c=0.0001):
        """
        Re-creation of the structure loss proposed in "Cycle Structure and Illumination
        Constrained GAN for Medical Image Enhancement" by Ma et al.
        :param images: List of original images
        :param generated_images: List of generated images (same order as originals)
        :param patch_size: Patch size to cut the images into (non-overlapping)
        :param c: Small positive constant to avoid errors for identical images
        """
        assert not c < 0
        structure_losses = []
        loss_n = 0
        for img, gen_img in zip(images, generated_images):
            img_patches, img_patches_number = create_patches(img, patch_size=patch_size)
            gen_img_patches, gen_img_patches_number = create_patches(gen_img, patch_size=patch_size)
            assert img_patches_number == gen_img_patches_number
            layers = img_patches.shape[3]
            covariances = []
            # Calculate the covariances between all patches
            for img_patch, gen_img_patch in zip(img_patches, gen_img_patches):
                # Calculate the covariance for the individual color layers
                cov_of_patch = np.empty([layers])
                for idx in range(layers):
                    combined = np.vstack((img_patch[:, :, idx].flatten(), gen_img_patch[:, :, idx].flatten()))
                    cov_matrix = np.cov(combined)
                    cov_of_patch[idx] = cov_matrix[0][1]
                covariances.append(cov_of_patch)
            # Calculate the standard deviations of the original image patches and the geneated images
            img_stds = np.std(img_patches, axis=(1, 2))
            gen_img_stds = np.std(gen_img_patches, axis=(1, 2))
            covariances = np.array(covariances)
            # Calculate the structure loss (included the number of layers,
            # which is not included in the definition in the paper)
            structure_loss = 1 - 1 / img_patches_number * 1 / layers * np.sum(
                (covariances + c) / (img_stds * gen_img_stds + c))
            # Make sure to stay within the boundaries since
            # the value is slightly negative for identical images (due to c)
            structure_loss = structure_loss.clip(0, 1)
            structure_losses.append(structure_loss)
            loss_n += 1
        structure_loss = np.sum(np.array(structure_losses)) / loss_n
        lambda_id = self.additional_losses.index("ma_structure_loss")
        return self.lambda_additional_losses[lambda_id] * structure_loss

    def test_saving(self, abs_gen_val_loss, epoch):
        """
        Test whether the early stopping criterion is fulfilled
        :param abs_gen_val_loss: Validation loss of the generator as summed up absolute sub-losses
        :param epoch: Current epoch of training
        :return: None
        """
        if abs_gen_val_loss < self.best_ckpt_loss:
            # Update best loss
            self.best_ckpt_loss = abs_gen_val_loss
            # Save the current states of the models + optimizers in a checkpoint
            ckpt_save_path = self.ckpt_manager.save(checkpoint_number=epoch)
            print('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))

    def train(self, training_file, validation_file, domain_to_translate):
        """
        Train a starGAN network on the current dataset consisting of two domain sets
        :param training_file: CSV file containing info on the training images
        :param validation_file: CSV file containing info on the validation images
        :param domain_to_translate: Name of the domain to translate from for visualization
        :return: None
        """
        # Unpack dataset
        train_files = get_filenames(training_file, self.domains)
        train_files_merged = get_filenames(training_file, self.domains, merge=True)
        val_files = get_filenames(validation_file, self.domains)
        # Calculate the number of batches per training epoch
        train_number = min(map(len, train_files))
        print("Maximum images per domain:", train_number)
        bat_per_epo = int(train_number / self.n_batch)
        # bat_per_epo = int(len(train_files) / self.n_batch)
        print(bat_per_epo, "updates per epoch")
        # Visualize the output of the generators before training
        self.visualize_performance(train_files, domain_to_translate, epoch='init')
        # Start training
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            print("Starting new epoch")
            # Update the learning rate of the optimizers depending on the current epoch
            self.update_lr(epoch)
            # Randomize order of training data
            for domain in train_files:
                random.shuffle(domain)
            train_files_this_epoch = cut_out_files(train_files, train_number)
            random.shuffle(train_files_this_epoch)
            # Empty lists for losses of each batch in one epoch
            self.training_logger.reset_batch()
            # Current time for displaying how long the epoch took
            start = time.time()
            for update in range(bat_per_epo):
                # Get a number of training images
                real_a, domains_a = get_real_samples(train_files_this_epoch, self.n_batch, update, domains=self.domains,
                                                     return_domain_names=True, all_files=train_files)
                # Labels to translate from
                labels_a = self.domains_to_labels(domains_a)
                # Randomly select which domains to transfer to
                domains_b = self.get_random_domains_without_original(self.n_batch, domains_a)
                # Labels to translate to
                labels_b = self.domains_to_labels(domains_b)
                # Persistent is set to True because the tape is used more than once to calculate the gradients.
                with tf.GradientTape(persistent=True) as tape:
                    # Generator fake images of the domains to transfer to
                    fake_b = self.generator([real_a, labels_b], training=True)

                    # Get results of the discriminator for real and fake data
                    disc_real_a = self.discriminator(real_a, training=True)
                    disc_fake_b = self.discriminator(fake_b, training=True)

                    # -------------ADVERSARIAL LOSS--------------------------:
                    # Loss indicating whether the discriminator was able to tell fake from real images
                    # Adversarial loss of the generator depends on whether the discriminator could tell
                    # that the generated images aren't real
                    gen_adv_loss = self.generator_adv_loss(prob_fake_is_real=disc_fake_b[0])
                    # Adversarial loss of the discriminator is a combination of the loss on real and fake data
                    discriminator_adv_loss = self.discriminator_adv_loss(prob_real_is_real=disc_real_a[0],
                                                                         prob_fake_is_real=disc_fake_b[0])

                    if not self.lambda_gradient_penalty == 0:
                        # --------------WGAN - gradient penalty loss------------:
                        # Only relevant for the discriminator
                        gradient_penalty = self.wgan_gp_loss(real_a, fake_b)
                    else:
                        gradient_penalty = 0

                    # -------------DOMAIN CLASSIFICATION LOSS----------------:
                    # The domain classification loss indicates whether the discriminator could correctly tell the domain
                    # the real and fake images are supposed to belong to, the loss on the real data is used for the
                    # discriminator and on the fake data on the generator
                    domain_class_real_loss = self.domain_class_loss(real_label=labels_a, predicted_label=disc_real_a[1])
                    domain_class_fake_loss = self.domain_class_loss(real_label=labels_b, predicted_label=disc_fake_b[1])

                    # -------------RECONSTRUCTION LOSS-----------------------:
                    # Reconstruct original images by translating back to domain a
                    cycled_a = self.generator([fake_b, labels_a], training=True)
                    reconstruction_loss = self.calc_cycle_loss(real_a, cycled_a)

                    # ---------------CONDITIONAL IDENTITY LOSS----------------------------:
                    # Same-domain translation (ideally no change)
                    same_a = self.generator([real_a, labels_a], training=True)
                    # Get result of the discriminator
                    disc_same_a = self.discriminator(same_a, training=True)
                    # Adversarial loss of the generator on the same-domain translation
                    gen_adv_id_loss = self.generator_adv_loss(prob_fake_is_real=disc_same_a[0])
                    # Generate cycled images
                    cycled_same_a = self.generator([same_a, labels_a], training=True)
                    # Domain classification loss on the same-domain translation
                    domain_class_id_loss = self.domain_class_loss(real_label=labels_a, predicted_label=disc_same_a[1])
                    # Reconstruct original image
                    reconstruction_id_loss = self.calc_cycle_loss(real_a, cycled_same_a)
                    # Identity loss of the same-domain translation
                    id_loss = self.conditional_identity_loss(real_a, same_a)
                    cond_identity_loss = gen_adv_id_loss + domain_class_id_loss + reconstruction_id_loss + id_loss

                    # --------------ADDITIONAL LOSSES-------------------------:
                    if 'ms_ssim_loss' in self.additional_losses:
                        reconstruction_loss += self.ms_ssim_loss(real_a, cycled_a)
                    if 'add_identity_loss' in self.additional_losses and epoch < 20:
                        cond_identity_loss += self.additional_identity_loss(real_a, same_a, epoch, final_epoch=20)
                    if 'ma_structure_loss' in self.additional_losses:
                        structure_loss = self.ma_structure_loss(real_a, fake_b)
                    else:
                        structure_loss = 0
                    # -------------Total generator loss-----------------------:
                    # = adversarial loss + domain class loss + reconstruction loss + conditional identity loss
                    generator_loss = gen_adv_loss + domain_class_fake_loss + reconstruction_loss + cond_identity_loss \
                                     + structure_loss
                    # -------------Total discriminator loss-------------------:
                    # = adversarial loss + gradient penalty + class loss
                    discriminator_loss = discriminator_adv_loss + gradient_penalty + domain_class_real_loss

                # Calculate the gradients for the generator
                generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
                # Calculate the gradients for the discriminator
                discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

                # "We perform one generator update after five discriminator updates"
                # if update % 5 == 0:
                # Apply the gradients
                self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                             self.generator.trainable_variables))
                # Apply the gradients
                self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                                 self.discriminator.trainable_variables))
                # Save the losses per batch to sum up later
                losses = [discriminator_loss, discriminator_adv_loss, gradient_penalty,
                          domain_class_real_loss, generator_loss, gen_adv_loss,
                          domain_class_fake_loss, cond_identity_loss, reconstruction_loss]
                self.training_logger.log_batch(losses)
                # Show progress once in a while
                if update % 20 == 0:
                    print('.')
            print('Time taken for epoch {} is {} sec\n'.format(epoch, time.time() - start))

            # Add summary of losses (means of the whole batch) to sacred
            self.training_logger.log_to_ex(epoch, learning_rate=self.curr_lr)

            # Export generated images to show the progress
            self.visualize_performance(train_files_merged, domain_to_translate, epoch)

            # Test performance on validation set and add the results to sacred
            abs_generator_val_loss = self.validate(val_files, epoch)
            self.test_saving(abs_generator_val_loss, epoch)

    def validate(self, val_files, epoch=None):
        """
        Test the performance of a model on a validation data set
        :param val_files: Paths to validation files
        :param epoch: How many epochs have been trained so far
        :return: None
        """
        if not epoch:
            epoch = self.start_epoch
        bat_per_epo = int(len(val_files) / self.n_batch)
        val_number = min(map(len, val_files))
        # Randomly shuffle the validation files (in place)
        for domain in val_files:
            random.shuffle(domain)
        # Only use a (random) subset of the validation images each epoch so that domains with more validation images
        # are not overshadowing the smaller domains regarding the overall validation loss, which dictates which
        # epochs are saved as checkpoints
        val_files_this_epoch = cut_out_files(val_files, val_number)
        random.shuffle(val_files_this_epoch)

        # Reset the batch losses for a new epoch
        self.validation_logger.reset_batch()
        for update in range(bat_per_epo):
            # Get a number of training images
            real_a, domains_a = get_real_samples(val_files_this_epoch, self.n_batch, update,
                                                 domains=self.domains, return_domain_names=True,
                                                 all_files=val_files)
            # Labels to translate from
            labels_a = self.domains_to_labels(domains_a)
            # Randomly select which domains to transfer to
            domains_b = self.get_random_domains_without_original(self.n_batch, domains_a)
            # Labels to translate to
            labels_b = self.domains_to_labels(domains_b)

            # Generator fake images of the domains to transfer to
            fake_b = self.generator([real_a, labels_b], training=False)

            # Get results of the discriminator for real and fake data
            disc_real_a = self.discriminator(real_a, training=False)
            disc_fake_b = self.discriminator(fake_b, training=False)

            # -------------ADVERSARIAL LOSS--------------------------:
            # Loss indicating whether the discriminator was able to tell fake from real images
            # Adversarial loss of the generator depends on whether the discriminator could tell
            # that the generated images aren't real
            gen_adv_loss = self.generator_adv_loss(prob_fake_is_real=disc_fake_b[0])
            # Adversarial loss of the discriminator is a combination of the loss on real and fake data
            discriminator_adv_loss = self.discriminator_adv_loss(prob_real_is_real=disc_real_a[0],
                                                                 prob_fake_is_real=disc_fake_b[0])

            if not self.lambda_gradient_penalty == 0:
                # --------------WGAN - gradient penalty loss------------:
                # Only relevant for the discriminator
                gradient_penalty = self.wgan_gp_loss(real_a, fake_b)
            else:
                gradient_penalty = 0
            # -------------DOMAIN CLASSIFICATION LOSS----------------:
            # The domain classification loss indicates whether the discriminator could correctly tell the domain
            # the real and fake images are supposed to belong to, the loss on the real data is used for the
            # discriminator and on the fake data on the generator
            domain_class_real_loss = self.domain_class_loss(real_label=labels_a, predicted_label=disc_real_a[1])
            domain_class_fake_loss = self.domain_class_loss(real_label=labels_b, predicted_label=disc_fake_b[1])

            # -------------RECONSTRUCTION LOSS-----------------------:
            # Reconstruct original images by translating back to domain a
            cycled_a = self.generator([fake_b, labels_a], training=False)
            reconstruction_loss = self.calc_cycle_loss(real_a, cycled_a)

            # ---------------CONDITIONAL IDENTITY LOSS----------------------------:
            # Same-domain translation (ideally no change)
            same_a = self.generator([real_a, labels_a], training=False)
            # Get result of the discriminator
            disc_same_a = self.discriminator(same_a, training=False)
            # Adversarial loss of the generator on the same-domain translation
            gen_adv_id_loss = self.generator_adv_loss(prob_fake_is_real=disc_same_a[0])
            # Generate cycled images
            cycled_same_a = self.generator([same_a, labels_a], training=False)
            # Domain classification loss on the same-domain translation
            domain_class_id_loss = self.domain_class_loss(real_label=labels_a, predicted_label=disc_same_a[1])
            # Reconstruct original image
            reconstruction_id_loss = self.calc_cycle_loss(real_a, cycled_same_a)
            # Identity loss of the same-domain translation
            id_loss = self.conditional_identity_loss(real_a, same_a)
            cond_identity_loss = gen_adv_id_loss + domain_class_id_loss + reconstruction_id_loss + id_loss

            # --------------ADDITIONAL LOSSES-------------------------:
            if 'ms_ssim_loss' in self.additional_losses:
                reconstruction_loss += self.ms_ssim_loss(real_a, cycled_a)
            if 'add_identity_loss' in self.additional_losses and epoch < 20:
                cond_identity_loss += self.additional_identity_loss(real_a, same_a, epoch, final_epoch=20)
            if 'ma_structure_loss' in self.additional_losses:
                structure_loss = self.ma_structure_loss(real_a, fake_b)
            else:
                structure_loss = 0
            # -------------Total generator loss-----------------------:
            # = adversarial loss + domain class loss + reconstruction loss + conditional identity loss
            generator_loss = gen_adv_loss + domain_class_fake_loss + reconstruction_loss + cond_identity_loss \
                             + structure_loss
            # -------------Total discriminator loss-------------------:
            # = adversarial loss + gradient penalty + class loss
            discriminator_loss = discriminator_adv_loss + gradient_penalty + domain_class_real_loss
            losses = [discriminator_loss, discriminator_adv_loss, gradient_penalty,
                      domain_class_real_loss, generator_loss, gen_adv_loss,
                      domain_class_fake_loss, cond_identity_loss, reconstruction_loss]
            self.validation_logger.log_batch(losses)
        # Add validation summary to sacred
        self.validation_logger.log_to_ex(epoch)
        # Get generator validation losses for early stopping
        mean_val_losses = self.validation_logger.get_batch_mean()
        absolute_generator_loss = abs(mean_val_losses[5]) + abs(mean_val_losses[6]) \
                                  + abs(mean_val_losses[7]) + abs(mean_val_losses[8])
        return absolute_generator_loss

    def transform_images(self, files, original_domain, domain_to_translate_to, val_test):
        """
        Transform images using the generators
        :param files: Numpy array containing all paths of files to transform
        :param original_domain: Domain to translate from
        :param domain_to_translate_to: Which domain the images are transformed to
        :param val_test: Whether the images are used for validating or testing
        :return: None
        """
        str_epoch = str(self.start_epoch)
        label_out = self.domains_to_labels(np.array([domain_to_translate_to]))
        for ix, path in enumerate(files):
            # Cut off path
            filename = path.split(settings.DB_DIR)[1]
            # Cut off filename
            path_to_file, filename = filename.rsplit('/', 1)
            path_sample_out = settings.OUTPUT_DIR + "/generated_images/" + self.run_id + "/" + str_epoch + "/" + \
                              "to_" + domain_to_translate_to + "/" + path_to_file + "/"
            if not os.path.exists(path_sample_out):
                os.makedirs(path_sample_out)
            if not (os.path.isfile(path_sample_out + filename)):
                # and os.path.isfile(path_cycled_out + filename)):
                print("Reading in image")
                sample_fullsize, sample, original_size = get_sample_from_path(path)
                sample_out = self.generator([sample, label_out], training=False)
                print("Upsampling generated image")
                sample_out_upsampled = laplacian_upsampling(originals=sample_fullsize.numpy(),
                                                            inputs=sample_out.numpy(),
                                                            original_shape=original_size)
                # Remove batch dimension
                sample_out_upsampled = np.squeeze(sample_out_upsampled)
                # Normalize for display
                sample_out_upsampled = normalize_for_display(sample_out_upsampled)
                print(path_sample_out + filename)
                if not os.path.isfile(path_sample_out + filename):
                    matplotlib.image.imsave(path_sample_out + filename, sample_out_upsampled)

    def evaluate(self, csv_file, val_test, domain_a, dataset):
        """
        Evaluate a loaded model by inspecting the transformed and cycled images
        :param csv_file: CSV file containing info on the images
        :param val_test: Whether the current evaluation is for validation or testing
        :param domain_a: Domain to transform
        :param dataset: Name of the dataset
        """
        epoch = self.start_epoch
        str_epoch = str(epoch)
        run_id = self.run_id

        # Prepare file names
        files = get_filenames(csv_file, self.domains)
        a_id = self.domains.index(domain_a)
        files_a = files[a_id]

        # Get target domains
        target_domains = deepcopy(self.domains)
        target_domains.remove(domain_a)
        print("Target domains:", target_domains)

        # Generate transformed and cycled images if they don't exist already
        paths_out = []
        for domain_b in target_domains:
            path_out = settings.OUTPUT_DIR + "/generated_images/" + run_id + "/" + str_epoch + "/" + "to_" + \
                       domain_b + "/"
            paths_out.append(path_out)
            self.transform_images(files_a, original_domain=domain_a, domain_to_translate_to=domain_b, val_test=val_test)

        for idx, domain_b in enumerate(target_domains):
            if val_test == 'validate':
                self.eval_val_logger.reset_batch()
            else:
                self.eval_test_logger.reset_batch()

            print("Evaluating transformation to", domain_b)
            for ix, path in enumerate(files_a):
                filename = path.split(settings.DB_DIR)[1]
                # Read in images in full size, remove batch dimension
                original_fullsize = np.squeeze(get_sample_from_path(path)[0])
                transformed_upsampled = np.squeeze(get_sample_from_path(paths_out[idx] + filename)[0])
                original_fullsize = normalize_for_evaluation(original_fullsize)
                transformed_upsampled = normalize_for_evaluation(transformed_upsampled)
                # Get the SSIM scores between input and output of the generators
                ssim_inout = ssim_score(original_fullsize, transformed_upsampled)
                if val_test == 'validate':
                    self.eval_val_logger.log_specific_batch([ssim_inout], ids=[0])
                else:
                    self.eval_test_logger.log_specific_batch([ssim_inout], ids=[0])
                print("Completed {}/{}".format(ix + 1, len(files_a)))

            # Read in all images again for FID
            b_id = self.domains.index(domain_b)
            files_b = files[b_id]

            print("Reading in images")
            # Get full paths of transformed images
            a_fullsize = get_all_samples(files_a)
            b_fullsize = get_all_samples(files_b)
            files_transformed_a = [os.path.join(path, name) for path, subdirs, files in os.walk(paths_out[idx])
                                   for name in files]
            to_b_upsampled = get_all_samples(files_transformed_a)

            # FID score:
            print("Calculating FID score between real domains and generated domains")
            fid_orig = get_fid(a_fullsize, b_fullsize)
            fid_b = get_fid(b_fullsize, to_b_upsampled)

            # Add summary to sacred
            if val_test == 'validate':
                self.eval_val_logger.log_specific_batch([fid_orig, fid_b], ids=[1, 2])
                self.eval_val_logger.log_to_ex(epoch)
                # Add validation summary to csv file
                means = self.eval_val_logger.get_batch_mean()
                save_to_csv(self.run_id, epoch, domain_a, domain_b, approach="fixedpointgan", dataset=dataset,
                            means=means, validate=True, only_ab=True)

            else:
                self.eval_test_logger.log_specific_batch([fid_orig, fid_b], ids=[1, 2])
                self.eval_test_logger.log_to_ex(epoch)

                # Add test summary to csv file
                means = self.eval_test_logger.get_batch_mean()
                save_to_csv(self.run_id, epoch, domain_a, domain_b, approach="fixedpointgan", dataset=dataset,
                            means=means, validate=False, only_ab=True)

