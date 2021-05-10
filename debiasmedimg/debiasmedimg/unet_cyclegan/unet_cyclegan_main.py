import os
from os import listdir
import time
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import debiasmedimg.settings as settings
from debiasmedimg.cyclegan.util import normalize_for_display, normalize_for_evaluation, get_filenames, \
    get_filtered_filenames, get_sample_from_path, get_real_samples, laplacian_upsampling, save_to_csv, Logger, \
    define_discriminator, ssim_score, get_fid, get_all_samples, create_patches
from .util import define_unet_generator


class UnetCycleGAN:
    """
    A class that encapsulates a cycleGAN implementation
    The losses used for training depend on  given parameters during the initialization
    """

    def __init__(self, ex, domain_names, load_epoch, run_id, image_shape, epochs, base_lr, lambda_adversarial_loss,
                 lambda_cycleloss,
                 lambda_identityloss, lambda_discriminator_loss, n_batch, n_resnet, additional_losses,
                 lambda_additional_losses):
        """
        Create a U-Net CycleGAN
        :param ex: Sacred experiment to log to
        :param domain_names: Names of the domains to transform between
        :param load_epoch: Which episode to load if run id already exists
        :param run_id: ID of the model if a model is loaded else None
        :param image_shape: Shape of the input image
        :param epochs: Number of epochs to train
        :param base_lr: Learning rate to start training with
        :param lambda_adversarial_loss: Lambda of the adversarial loss
        :param lambda_cycleloss: Lambda of the cycle loss
        :param lambda_identityloss: Lambda of the identity loss
        :param n_batch: Number of training samples per batch
        :param n_resnet: Number of resNet blocks in the generator
        :param additional_losses: Which additional losses to use for training
        :param lambda_additional_losses: How to weigh the additional losses
        """
        # Save parameters
        self.ex = ex
        self.domains = domain_names
        self.load_epoch = load_epoch
        self.run_id = run_id
        self.image_shape = image_shape
        self.epochs = epochs
        self.base_lr = base_lr
        self.lambda_adversarial_loss = lambda_adversarial_loss
        self.lambda_cycleloss = lambda_cycleloss
        self.lambda_identityloss = lambda_identityloss
        self.n_batch = n_batch
        self.n_resnet = n_resnet
        self.lambda_discriminator_loss = lambda_discriminator_loss
        self.additional_losses = additional_losses
        self.lambda_additional_losses = lambda_additional_losses

        # Create run-id
        if not self.run_id:
            self.run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.run_id = self.run_id

        # Initialize learning rate
        self.curr_lr = self.base_lr

        # Create models
        self.generator_AtoB = define_unet_generator(self.image_shape, self.n_resnet)
        self.generator_BtoA = define_unet_generator(self.image_shape, self.n_resnet)
        # A -> real/fake
        self.discriminator_A = define_discriminator(self.image_shape)
        # B -> real/fake
        self.discriminator_B = define_discriminator(self.image_shape)

        # Create one optimizer per model
        self.generator_AtoB_optimizer = tf.keras.optimizers.Adam(self.base_lr, beta_1=0.5)
        self.generator_BtoA_optimizer = tf.keras.optimizers.Adam(self.base_lr, beta_1=0.5)
        self.discriminator_A_optimizer = tf.keras.optimizers.Adam(self.base_lr, beta_1=0.5)
        self.discriminator_B_optimizer = tf.keras.optimizers.Adam(self.base_lr, beta_1=0.5)

        # Create checkpoint manager for saving the models during training
        self.checkpoint_path = settings.OUTPUT_DIR + "/checkpoints/" + self.run_id + "/train"
        # Define what to store in the checkpoint
        self.ckpt = tf.train.Checkpoint(generator_AtoB=self.generator_AtoB,
                                        generator_BtoA=self.generator_BtoA,
                                        discriminator_A=self.discriminator_A,
                                        discriminator_B=self.discriminator_B,
                                        generator_AtoB_optimizer=self.generator_AtoB_optimizer,
                                        generator_BtoA_optimizer=self.generator_BtoA_optimizer,
                                        discriminator_A_optimizer=self.discriminator_A_optimizer,
                                        discriminator_B_optimizer=self.discriminator_B_optimizer)
        # max_to_keep = None save all checkpoints
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)

        # If a checkpoint exists, restore the latest checkpoint
        if self.ckpt_manager.latest_checkpoint:
            if load_epoch:
                ckpt_to_restore = [s for s in self.ckpt_manager.checkpoints if "ckpt-" + str(self.load_epoch) in s][0]
            else:
                ckpt_to_restore = self.ckpt_manager.latest_checkpoint
            print(ckpt_to_restore)
            status = self.ckpt.restore(ckpt_to_restore).expect_partial()
            print('Latest checkpoint restored!!')

            # The number of the checkpoint indicates how many epochs have been trained so far
            path, id_and_checkpoint = ckpt_to_restore.split('checkpoints/')
            self.start_epoch = int(id_and_checkpoint.split('-')[2])
            status.assert_existing_objects_matched()
        else:
            self.start_epoch = 0

        # Initiate empty batch losses
        losses = ["discriminator_a_loss", "discriminator_b_loss", "generator_ab_loss", "generator_ba_loss",
                  "identity_ab_loss", "identity_ba_loss", "adversarial_ab_loss", "adversarial_ba_loss",
                  "total_cycle_loss"]
        evaluation_metrics = ["ssim_inout_a", "ssim_inout_b", "fid_orig", "fid_a", "fid_b"]
        self.training_logger = Logger(ex, losses, mode='train')
        self.validation_logger = Logger(ex, losses, mode='validate')
        self.eval_val_logger = Logger(ex, evaluation_metrics, mode='evaluate_val')
        self.eval_test_logger = Logger(ex, evaluation_metrics, mode='evaluate_test')

        # Initiate for visualization of training
        self.vis_a_img = None
        self.vis_b_img = None
        # Initiate for identifying best training epoch
        self.best_gen_ab_val_loss = float('Inf')
        self.best_gen_ba_val_loss = float('Inf')

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
        self.generator_AtoB_optimizer.lr.assign(self.curr_lr)
        self.generator_BtoA_optimizer.lr.assign(self.curr_lr)
        self.discriminator_A_optimizer.lr.assign(self.curr_lr)
        self.discriminator_B_optimizer.lr.assign(self.curr_lr)
        print("Current learning rate:", self.generator_AtoB_optimizer.lr.numpy())

    def generate_fake_samples(self, g_model, data, patch_shape):
        """
        Generate a batch of fake images, returns images and targets
        :param g_model: String indicating which generator is generating samples
        :param data: Data to predict based on
        :param patch_shape: Shape of the patch
        :return: Fake images and targets
        """
        # generate fake instance
        x = None
        if g_model == 'BtoA':
            x = self.generator_BtoA.predict(data)
        elif g_model == 'AtoB':
            x = self.generator_AtoB.predict(data)
        else:
            print("Wrong input")
            exit()
        # create 'fake' class labels (0)
        y = np.zeros((len(x), patch_shape, patch_shape, 1))
        return x, y

    def visualize_performance(self, train_files_a, train_files_b, epoch):
        """
        Put out a plot showing how the generators transform the images after each epoch
        :param train_files_a: Training set A
        :param train_files_b: Training set B
        :param epoch: Current epoch of the training process
        :return:
        """
        if epoch == 'init':
            # Before training starts decide on an image and use it for visualization through the whole process
            self.vis_a_img = train_files_a[0]
            self.vis_b_img = train_files_b[0]
        # Prepare images
        _, sample_a, _ = get_sample_from_path(self.vis_a_img)
        _, sample_b, _ = get_sample_from_path(self.vis_b_img)
        # Get transformed images from generators
        to_b = self.generator_AtoB(sample_a)
        to_a = self.generator_BtoA(sample_b)
        # Normalize images to [0,1]
        sample_a = normalize_for_display(sample_a)
        sample_b = normalize_for_display(sample_b)
        to_b = normalize_for_display(to_b)
        to_a = normalize_for_display(to_a)
        # Create figure for displaying images
        plt.figure(figsize=(8, 8))
        images = [sample_a, to_b, sample_b, to_a]
        title = ['A', 'A to B', 'B', 'B to A']
        for i in range(len(images)):
            plt.subplot(2, 2, i + 1)
            plt.title(title[i])
            plt.imshow(images[i][0])
        path = settings.OUTPUT_DIR + "/plots/" + self.run_id + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + str(epoch) + ".png")
        plt.close()

    def discriminator_loss(self, prob_real_is_real, prob_fake_is_real):
        """
        Loss of the discriminator
        :param prob_real_is_real: Loss of the discriminator on real samples
        :param prob_fake_is_real: Loss of the discriminator on fake samples
        :return: total loss of the discriminator divided by two
        """
        # p(real/fake) = 1 if it is real, 0 if it is fake
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_is_real_loss = loss_obj(tf.ones_like(prob_real_is_real), prob_real_is_real)
        fake_is_real_loss = loss_obj(tf.zeros_like(prob_fake_is_real), prob_fake_is_real)
        # The loss of D is divided by half (loss_weights) to slow down the updates of the discriminator
        return self.lambda_discriminator_loss * (real_is_real_loss + fake_is_real_loss)

    def calc_cycle_loss(self, real_image, cycled_image):
        """
        Calculate the loss between the real input and the cycled output (ideally identical)
        :param real_image: Real input
        :param cycled_image: Cycled output generated from the real input
        :return: cycle loss of the network
        """
        loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.lambda_cycleloss * loss

    def generator_loss(self, prob_fake_is_real):
        """
        Calculate the generator loss (whether the discriminator was able to tell fake from real images)
        :param prob_fake_is_real: Probabilities predicted by the discriminator
        :return: generator loss
        """
        # p(real/fake) = 1 if it is real, 0 if it is fake
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = loss_obj(tf.ones_like(prob_fake_is_real), prob_fake_is_real)
        return self.lambda_adversarial_loss * loss

    def identity_loss(self, real_image, same_image):
        """
        Calculate the identity loss between the real image and the transformed version (ideally identical)
        :param real_image: Real input
        :param same_image: Transformed input
        :return: Identity loss
        """
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
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
        lambda_add_identity_loss = self.lambda_additional_losses[lambda_id] \
            - epoch * self.lambda_additional_losses[lambda_id] / final_epoch
        return lambda_add_identity_loss * loss

    def ms_ssim_loss(self, image_a, cycled_image_a, image_b, cycled_image_b):
        """
        MS-SSIM loss as used by Armanious et al. for MR images
        :param image_a: Image of domain A
        :param cycled_image_a: Cycled image of domain A
        :param image_b: Image of domain B
        :param cycled_image_b: Cycled image of domain B
        :return: MS-SSMI loss
        """
        # max-val = difference between the maximum the and minimum allowed values,
        # images here are normalized to be in range [-1,1] -> max_val = 2
        ms_ssim = (1 - tf.image.ssim_multiscale(image_a, cycled_image_a, max_val=2)) + \
                  (1 - tf.image.ssim_multiscale(image_b, cycled_image_b, max_val=2))
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

    def train(self, training_file, validation_file):
        """
        Train a cycleGAN network on the current dataset consisting of two domain sets
        :param training_file: CSV file containing info on the training images
        :param validation_file: CSV file containing info on the validation images
        :return: None
        """
        # Unpack dataset
        train_files = get_filenames(training_file, self.domains)
        print("Number of domains:", len(self.domains))
        print("Domain names:", self.domains)
        train_files_a = train_files[0]
        train_files_b = train_files[1]
        val_files = get_filenames(validation_file, self.domains)
        val_files_a = val_files[0]
        val_files_b = val_files[1]

        # Calculate the number of batches per training epoch
        train_length = min(len(train_files_a), len(train_files_b))
        bat_per_epo = int(train_length / self.n_batch)
        print(bat_per_epo, "updates per epoch")

        # Visualize the output of the generators before training
        self.visualize_performance(train_files_a, train_files_b, epoch='init')

        # Start training
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            print("Starting new epoch")
            # Shuffle training data at the beginning of an epoch
            np.random.shuffle(train_files_a)
            np.random.shuffle(train_files_b)

            # Update the learning rate of the optimizers depending on the current epoch number
            self.update_lr(epoch)

            # Empty lists for losses of each batch in one epoch
            self.training_logger.reset_batch()

            # Current time for displaying how long the epoch took
            start = time.time()

            for update in range(bat_per_epo):
                # Select a batch of real samples
                real_a = get_real_samples(train_files_a, self.n_batch, update, self.domains)
                real_b = get_real_samples(train_files_b, self.n_batch, update, self.domains)

                # Persistent is set to True because the tape is used more than once to calculate the gradients.
                with tf.GradientTape(persistent=True) as tape:
                    # Forward cycle
                    fake_b = self.generator_AtoB(real_a, training=True)
                    cycled_a = self.generator_BtoA(fake_b, training=True)

                    # Backward cycle
                    fake_a = self.generator_BtoA(real_b, training=True)
                    cycled_b = self.generator_AtoB(fake_a, training=True)

                    # Get results of the discriminators for real and fake data
                    disc_real_a = self.discriminator_A(real_a, training=True)
                    disc_real_b = self.discriminator_B(real_b, training=True)
                    disc_fake_a = self.discriminator_A(fake_a, training=True)
                    disc_fake_b = self.discriminator_B(fake_b, training=True)

                    # ------------------ADVERSARIAL LOSS---------------------:
                    # Loss indicating whether the discriminator was able to tell fake from real images
                    adversarial_ab_loss = self.generator_loss(prob_fake_is_real=disc_fake_b)
                    adversarial_ba_loss = self.generator_loss(prob_fake_is_real=disc_fake_a)

                    # ------------------IDENTITY LOSS------------------------:
                    # Generator A to B should leave image of domain B unchanged and vice versa
                    same_b = self.generator_AtoB(real_b, training=True)
                    same_a = self.generator_BtoA(real_a, training=True)
                    identity_ab_loss = self.identity_loss(real_b, same_b)
                    identity_ba_loss = self.identity_loss(real_a, same_a)

                    # ------------------CYCLE-LOSS---------------------------:
                    # forward + backward
                    total_cycle_loss = self.calc_cycle_loss(real_a, cycled_a) + self.calc_cycle_loss(real_b, cycled_b)

                    # ------------------ADDITIONAL-LOSSES---------------------------:
                    if 'add_identity_loss' in self.additional_losses and epoch < 20:
                        identity_ab_loss += self.additional_identity_loss(real_b, same_b, epoch, final_epoch=20)
                        identity_ba_loss += self.additional_identity_loss(real_a, same_a, epoch, final_epoch=20)
                    if 'ms_ssim_loss' in self.additional_losses:
                        total_cycle_loss += self.ms_ssim_loss(real_a, cycled_a, real_b, cycled_b)
                    if 'ma_structure_loss' in self.additional_losses:
                        structure_loss = self.ma_structure_loss(real_a, fake_b)
                        structure_loss += self.ma_structure_loss(real_b, fake_a)
                    else:
                        structure_loss = 0
                    # -----------------Total generator loss------------------:
                    # = adversarial loss + cycle loss + identity loss
                    generator_ab_loss = adversarial_ab_loss + total_cycle_loss + identity_ab_loss + structure_loss
                    generator_ba_loss = adversarial_ba_loss + total_cycle_loss + identity_ba_loss + structure_loss

                    # -----------------Total discriminator loss--------------:
                    # Loss of the discriminator is a combination of the loss on real and fake data
                    discriminator_a_loss = self.discriminator_loss(prob_real_is_real=disc_real_a,
                                                                   prob_fake_is_real=disc_fake_a)
                    discriminator_b_loss = self.discriminator_loss(prob_real_is_real=disc_real_b,
                                                                   prob_fake_is_real=disc_fake_b)

                # Calculate the gradients for generator and discriminator
                generator_g_gradients = tape.gradient(generator_ab_loss,
                                                      self.generator_AtoB.trainable_variables)
                generator_f_gradients = tape.gradient(generator_ba_loss,
                                                      self.generator_BtoA.trainable_variables)
                discriminator_a_gradients = tape.gradient(discriminator_a_loss,
                                                          self.discriminator_A.trainable_variables)
                discriminator_b_gradients = tape.gradient(discriminator_b_loss,
                                                          self.discriminator_B.trainable_variables)

                # Apply the gradients to the optimizer
                self.generator_AtoB_optimizer.apply_gradients(zip(generator_g_gradients,
                                                                  self.generator_AtoB.trainable_variables))
                self.generator_BtoA_optimizer.apply_gradients(zip(generator_f_gradients,
                                                                  self.generator_BtoA.trainable_variables))
                self.discriminator_A_optimizer.apply_gradients(zip(discriminator_a_gradients,
                                                                   self.discriminator_A.trainable_variables))
                self.discriminator_B_optimizer.apply_gradients(zip(discriminator_b_gradients,
                                                                   self.discriminator_B.trainable_variables))

                # Save the losses per batch to sum up later
                losses = discriminator_a_loss, discriminator_b_loss, generator_ab_loss, generator_ba_loss, \
                         identity_ab_loss, identity_ba_loss, adversarial_ab_loss, adversarial_ba_loss, total_cycle_loss
                self.training_logger.log_batch(losses)
                # Show progress once in a while
                if update % 20 == 0:
                    print('.')
            print('Time taken for epoch {} is {} sec\n'.format(epoch, time.time() - start))

            # Add summary of losses (means of the whole batch) to sacred
            self.training_logger.log_to_ex(epoch=epoch, learning_rate=self.curr_lr)

            # Export generated images to show the progress
            self.visualize_performance(train_files_a, train_files_b, epoch)

            # Test performance on validation set and add the results to sacred
            generator_ab_val_loss, generator_ba_val_loss = self.validate(val_files_a, val_files_b, epoch)

            # Only save checkpoint if it has a better performance regarding the validation loss of the generator
            # transforming from b to a to save disk space
            flag_save_checkpoint = False
            # If the current loss is better than the best one so far update best losses
            if generator_ab_val_loss < self.best_gen_ab_val_loss:
                self.best_gen_ab_val_loss = generator_ab_val_loss
                # We care more about generating b from a
                flag_save_checkpoint = True
            if generator_ba_val_loss < self.best_gen_ba_val_loss:
                self.best_gen_ba_val_loss = generator_ba_val_loss
            if flag_save_checkpoint:
                # Save the current states of the models + optimizers in a checkpoint
                ckpt_save_path = self.ckpt_manager.save(checkpoint_number=epoch)
                print('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))

    def validate(self, val_files_a, val_files_b, epoch=None):
        """
        Test the performance of a model on a validation data set
        :param val_files_a: Validation files of domain A
        :param val_files_b: Validation files of domain B
        :param epoch: How many epochs have been trained so far
        :return: generator losses
        """
        if not epoch:
            epoch = self.start_epoch
        # Calculate the number of batches per validation
        val_length = min(len(val_files_a), len(val_files_b))
        bat_per_epo = int(val_length / self.n_batch)

        # Randomly shuffle the validation files
        np.random.shuffle(val_files_a)
        np.random.shuffle(val_files_b)

        self.validation_logger.reset_batch()

        for update in range(bat_per_epo):
            # Select a batch of real samples
            real_a = get_real_samples(val_files_a, self.n_batch, update, self.domains)
            real_b = get_real_samples(val_files_b, self.n_batch, update, self.domains)

            # Forward cycle
            fake_b = self.generator_AtoB(real_a, training=False)
            cycled_a = self.generator_BtoA(fake_b, training=False)

            # Backward cycle
            fake_a = self.generator_BtoA(real_b, training=False)
            cycled_b = self.generator_AtoB(fake_a, training=False)

            # Get results of the discriminators for real and fake data
            disc_real_a = self.discriminator_A(real_a, training=False)
            disc_real_b = self.discriminator_B(real_b, training=False)
            disc_fake_a = self.discriminator_A(fake_a, training=False)
            disc_fake_b = self.discriminator_B(fake_b, training=False)

            # ------------------ADVERSARIAL LOSS---------------------:
            # Loss indicating whether the discriminator was able to tell fake from real images
            adversarial_ab_loss = self.generator_loss(prob_fake_is_real=disc_fake_b)
            adversarial_ba_loss = self.generator_loss(prob_fake_is_real=disc_fake_a)

            # ------------------IDENTITY LOSS------------------------:
            # Generator A to B should leave image of domain B unchanged and vice versa
            same_b = self.generator_AtoB(real_b, training=False)
            same_a = self.generator_BtoA(real_a, training=False)
            identity_ab_loss = self.identity_loss(real_b, same_b)
            identity_ba_loss = self.identity_loss(real_a, same_a)

            # ------------------CYCLE-LOSS---------------------------:
            # forward + backward
            total_cycle_loss = self.calc_cycle_loss(real_a, cycled_a) + self.calc_cycle_loss(real_b, cycled_b)

            # ------------------ADDITIONAL-LOSSES---------------------------:
            if 'add_identity_loss' in self.additional_losses and epoch < 20:
                identity_ab_loss += self.additional_identity_loss(real_b, same_b, epoch, final_epoch=20)
                identity_ba_loss += self.additional_identity_loss(real_a, same_a, epoch, final_epoch=20)
            if 'ms_ssim_loss' in self.additional_losses:
                total_cycle_loss += self.ms_ssim_loss(real_a, cycled_a, real_b, cycled_b)
            if 'ma_structure_loss' in self.additional_losses:
                structure_loss = self.ma_structure_loss(real_a, fake_b)
                structure_loss += self.ma_structure_loss(real_b, fake_a)
            else:
                structure_loss = 0
            # -----------------Total generator loss------------------:
            # = adversarial loss + cycle loss + identity loss
            generator_ab_loss = adversarial_ab_loss + total_cycle_loss + identity_ab_loss + structure_loss
            generator_ba_loss = adversarial_ba_loss + total_cycle_loss + identity_ba_loss + structure_loss

            # -----------------Total discriminator loss--------------:
            # Loss of the discriminator is a combination of the loss on real and fake data
            discriminator_a_loss = self.discriminator_loss(prob_real_is_real=disc_real_a,
                                                           prob_fake_is_real=disc_fake_a)
            discriminator_b_loss = self.discriminator_loss(prob_real_is_real=disc_real_b,
                                                           prob_fake_is_real=disc_fake_b)

            # Save the losses per batch to sum up later
            losses = discriminator_a_loss, discriminator_b_loss, generator_ab_loss, generator_ba_loss, \
                identity_ab_loss, identity_ba_loss, adversarial_ab_loss, adversarial_ba_loss, total_cycle_loss
            self.validation_logger.log_batch(losses)

        # Add validation summary to sacred
        self.validation_logger.log_to_ex(epoch)

        # Get generator validation losses for checkpoint saving
        mean_val_losses = self.validation_logger.get_batch_mean()
        # Return validation losses for generator a->b and b->a
        return mean_val_losses[2], mean_val_losses[3]

    def transform_images(self, csv_file, domain_to_translate, domain_to_translate_to, val_test):
        """
        Transform images using the generators
        :param csv_file: File containing info about the images
        :param domain_to_translate: Name of the domain which is supposed to be translated
        :param domain_to_translate_to: Which domain the images are transformed to ("A" or "B")
        :param val_test: Whether the images are transformed for validation or testing
        :return: None
        """
        files = get_filtered_filenames(csv_file, domain_to_translate)
        for ix, path in enumerate(files):
            # Cut off path
            filename = path.split(settings.DB_DIR)[1]
            # Cut off filename
            path_to_file, filename = filename.rsplit('/', 1)
            str_epoch = str(self.start_epoch)
            path_sample_out = settings.OUTPUT_DIR + "/generated_images/" + self.run_id + "/" + \
                str_epoch + "/" + "to_" + str(domain_to_translate_to) + "/" \
                + path_to_file + "/"
            if not os.path.exists(path_sample_out):
                os.makedirs(path_sample_out)
            if not (os.path.isfile(path_sample_out + filename)):
                print("Reading in image")
                sample_fullsize, sample, original_size = get_sample_from_path(path)
                if domain_to_translate_to == self.domains[1]:
                    sample_out = self.generator_AtoB(sample, training=False)
                elif domain_to_translate_to == self.domains[0]:
                    sample_out = self.generator_BtoA(sample, training=False)
                else:
                    print("Invalid target domain")
                    exit()
                print("Upsampling generated image")
                sample_out_upsampled = laplacian_upsampling(originals=sample_fullsize.numpy(),
                                                            inputs=sample_out.numpy(),
                                                            original_shape=original_size)
                # Remove batch dimension
                sample_out_upsampled = np.squeeze(sample_out_upsampled)
                # Normalize for display
                sample_out_upsampled = normalize_for_display(sample_out_upsampled)
                if not os.path.isfile(path_sample_out + filename):
                    matplotlib.image.imsave(path_sample_out + filename, sample_out_upsampled)

    def evaluate(self, csv_file, val_test, dataset, evaluation_direction=None):
        """
        Validate the performance of a model on a validation data set
        :param csv_file: CSV file describing the images used for the evaluation
        :param val_test: Whether we are currently validating or testing
        :param dataset: Name of the dataset
        :evaluation_direction: If None both directions are evaluated, otherwise only the given one: "AB" or "BA"
        :return: None
        """
        epoch = self.start_epoch
        run_id = self.run_id

        # Prepare file names
        files = get_filenames(csv_file, self.domains)
        files_a = files[0]
        files_b = files[1]
        # Generate transformed and cycled images if they don't exist already
        # Cut off path
        str_epoch = str(self.start_epoch)
        path_out_a = settings.OUTPUT_DIR + "/generated_images/" + run_id + "/" + str_epoch + "/" + "to_" + \
                     str(self.domains[1]) + "/"
        path_out_b = settings.OUTPUT_DIR + "/generated_images/" + run_id + "/" + str_epoch + "/" + "to_" + \
                     str(self.domains[0]) + "/"

        # Transform images
        if evaluation_direction is None or evaluation_direction == "AB":
            self.transform_images(csv_file, domain_to_translate=self.domains[0], domain_to_translate_to=self.domains[1],
                                  val_test=val_test)
        if evaluation_direction is None or evaluation_direction == "BA":
            self.transform_images(csv_file, domain_to_translate=self.domains[1], domain_to_translate_to=self.domains[0],
                                  val_test=val_test)

        # Prepare logger (validate or test)
        if val_test == 'validate':
            self.eval_val_logger.reset_batch()
        else:
            self.eval_test_logger.reset_batch()
        if evaluation_direction is None or evaluation_direction == "AB":
            print("Evaluating set a")
            for ix, path in enumerate(files_a):
                filename = path.split(settings.DB_DIR)[1]
                # Read in images in full size, remove batch dimension
                original_fullsize = np.squeeze(get_sample_from_path(path)[0])
                transformed_upsampled = np.squeeze(get_sample_from_path(path_out_a + filename)[0])
                original_fullsize = normalize_for_evaluation(original_fullsize)
                transformed_upsampled = normalize_for_evaluation(transformed_upsampled)
                # Get the SSIM scores between input and output of the generators
                ssim_inout = ssim_score(original_fullsize, transformed_upsampled)
                if val_test == 'validate':
                    self.eval_val_logger.log_specific_batch([ssim_inout], ids=[0])
                else:
                    self.eval_test_logger.log_specific_batch([ssim_inout], ids=[0])
                print("Completed {}/{}".format(ix + 1, len(files_a)))
        
        if evaluation_direction is None or evaluation_direction == "BA":
            print("Evaluating set b")
            for ix, path in enumerate(files_b):
                filename = path.split(settings.DB_DIR)[1]
                print(path)
                original_fullsize = np.squeeze(get_sample_from_path(path)[0])
                transformed_upsampled = np.squeeze(get_sample_from_path(path_out_b + filename)[0])
                original_fullsize = normalize_for_evaluation(original_fullsize)
                transformed_upsampled = normalize_for_evaluation(transformed_upsampled)
                # Get the SSIM scores between input and output of the generators
                ssim_inout = ssim_score(original_fullsize, transformed_upsampled)
                if val_test == 'validate':
                    self.eval_val_logger.log_specific_batch([ssim_inout], ids=[1])
                else:
                    self.eval_test_logger.log_specific_batch([ssim_inout], ids=[1])
                print("Completed {}/{}".format(ix + 1, len(files_b)))

        # Read in all images again for FID
        a_fullsize = get_all_samples(files_a)
        b_fullsize = get_all_samples(files_b)
        fid_orig = get_fid(a_fullsize, b_fullsize) 
        fid_a = -1
        fid_b = -1
        # FID score:
        print("Calculating FID score between real domains and generated domains")

        # Get full paths of transformed images
        if evaluation_direction is None or evaluation_direction == "AB":

            files_transformed_a = [os.path.join(path, name) for path, subdirs, files in os.walk(path_out_a) for name in
                               files]
            to_b_upsampled = get_all_samples(files_transformed_a)
            fid_b = get_fid(b_fullsize, to_b_upsampled)

        if evaluation_direction is None or evaluation_direction == "BA":

            files_transformed_b = [os.path.join(path, name) for path, subdirs, files in os.walk(path_out_b) for name in
                               files]
            to_a_upsampled = get_all_samples(files_transformed_b)
            fid_a = get_fid(a_fullsize, to_a_upsampled)


        # Add summary to sacred
        if val_test == 'validate':
            self.eval_val_logger.log_specific_batch([fid_orig, fid_a, fid_b], ids=[2, 3, 4])
            self.eval_val_logger.log_to_ex(epoch)
            # Add validation summary to csv file
            means = self.eval_val_logger.get_batch_mean()
            save_to_csv(self.run_id, epoch, self.domains[0], self.domains[1], means, "unet_cyclegan", dataset,
                        validate=True)
        else:
            self.eval_test_logger.log_specific_batch([fid_orig, fid_a, fid_b], ids=[2, 3, 4])
            self.eval_test_logger.log_to_ex(epoch)
            # Add test summary to csv file
            means = self.eval_test_logger.get_batch_mean()
            save_to_csv(self.run_id, epoch, self.domains[0], self.domains[1], means, "unet_cyclegan", dataset,
                        validate=False)
