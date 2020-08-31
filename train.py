# Created by pengsheng.huang at 8/14/2020

from model import Generator, invG, Discriminator, AE, VAEGAN, AAE, make_classifier
import tensorflow as tf
import numpy as np
import os
import config
import time
import matplotlib.pyplot as plt
from dataset import build_dataset
from config import TrainMode
import pandas as pd
import cv2

from absl import app, flags, logging
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_enum_class(
    'train_mode',
    default=TrainMode.ALL,
    enum_class=TrainMode,
    help="Select training mode, one of {'ALL', 'GAN', 'VAE'}"
)

# GAN functions.
seed = tf.random.normal([4, config.NZ])

generator = Generator().build()
discriminator = Discriminator().build()
inv_generator = invG().build()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_opt = tf.keras.optimizers.Adam(config.LR)
discriminator_opt = tf.keras.optimizers.Adam(config.LR)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def gan_train_step(images, idx):
    noise = tf.random.normal([config.BATCH_SIZE, config.NZ])

    with tf.GradientTape() as gen_type, tf.GradientTape() as disc_tap:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradient_of_generator = gen_type.gradient(gen_loss, generator.trainable_variables)
    gradient_of_discriminator = disc_tap.gradient(disc_loss, discriminator.trainable_variables)

    generator_opt.apply_gradients(zip(gradient_of_generator, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(gradient_of_discriminator, discriminator.trainable_variables))


def gan_generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(2, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(2, 2, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('output/gan_output/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


def gan_train(dataset, epochs):
    checkpoint_dir = './gan_training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_opt,
                                     discriminator_optimizer=discriminator_opt,
                                     generator=generator,
                                     discriminator=discriminator)

    for epoch in range(epochs):
        start = time.time()

        for idx, image_batch in enumerate(dataset):
            gan_train_step(image_batch)

        # plt.cla()
        gan_generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix + "_epoch_%d" % (epoch + 1))

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # plt.cla()
    gan_generate_and_save_images(generator,
                                 epochs,
                                 seed)

    checkpoint.save(file_prefix=checkpoint_prefix + "_final")

    gan_generate_and_save_images(generator,
                                 0,
                                 seed)


# vae functions


def ae_generate_and_save_images(model, epoch, test_input, path):
    predictions = model.represent(test_input)
    fig = plt.figure(figsize=(4, 2))
    plt.tight_layout()

    for i in range(predictions.shape[0]):
        plt.subplot(4, 2, 2 * i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.subplot(4, 2, 2 * i + 2)
        plt.imshow(test_input[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(path, 'image_at_epoch_{:04d}.png'.format(epoch)))
    # plt.show()


def ae_train(train_dataset, test_dataset, epochs, mode):
    # reload and freeze model.

    gan_checkpoint_dir = './gan_training_checkpoints'
    gan_checkpoint = tf.train.Checkpoint(
        optimizer=generator_opt,
        discriminator_optimizer=discriminator_opt,
        generator=generator,
        discriminator=discriminator,
    )

    latest = tf.train.latest_checkpoint(gan_checkpoint_dir)
    gan_checkpoint.restore(latest)

    frozen_generator = gan_checkpoint.generator
    frozen_generator.trainable = False

    # build vae model.
    vae_opt = tf.keras.optimizers.Adam(1e-3)

    if mode == TrainMode.AE:
        print("*********************** Training naive AE***************************")
        model = AE(frozen_generator)
        checkpoint_dir = './ae_training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(AE_optimizer=model.optimizer,
                                         AE=model)
        print("generator is trainable: ", model.generator_net.trainable)
        path = './output/ae_output'
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        print("*********************** Training Auxiliary AE***************************")
        model = AAE(gan_checkpoint)
        model.discriminator.trainable = False
        model.generator_net.trainable = False
        # model.build((128, 128))
        print(model.discriminator.summary())
        print(model.discriminator.get_layer('conv2d_3'))
        checkpoint_dir = './aae_training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(AE_optimizer=model.optimizer,
                                         AE=model)
        path = './output/aae_output'
        print("generator is trainable: ", model.generator_net.trainable)
        if not os.path.exists(path):
            os.makedirs(path)

    # train
    sample_input = next(iter(test_dataset))[:4, ::]
    ae_generate_and_save_images(model, 0, sample_input, path)
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            model.train(train_x)
        end_time = time.time()

        if epoch % 1 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(model.compute_loss(test_x))
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, '
                  'time elapse for current epoch {}'.format(epoch,
                                                            elbo,
                                                            end_time - start_time))
            ae_generate_and_save_images(
                model, epoch, sample_input, path)
            checkpoint.save(file_prefix=checkpoint_prefix + "_epoch_%d" % (epoch + 1))
        if epoch == epochs - 1:
            ae_generate_and_save_images(
                model, epoch, sample_input, path)
            checkpoint.save(file_prefix=checkpoint_prefix + "_epoch_%d" % (epoch + 1))
            # tf.saved_model.save(model, checkpoint_dir + "/saved_model")


# vae gan
def plot_reconstruction(model, example_data, path, epoch, nex=8, zm=2):
    example_data_reconstructed = model.reconstruct(example_data)
    samples = model.decode(tf.random.normal(shape=(config.BATCH_SIZE, config.NZ)))
    fig, axs = plt.subplots(ncols=nex, nrows=3, figsize=(zm * nex, zm * 3))
    for axi, (dat, lab) in enumerate(
            zip(
                [example_data, example_data_reconstructed, samples],
                ["data", "data recon", "samples"],
            )
    ):
        for ex in range(nex):
            axs[axi, ex].matshow(
                dat.numpy()[ex].squeeze(), cmap=plt.cm.Greys, vmin=0, vmax=1
            )
            axs[axi, ex].axes.get_xaxis().set_ticks([])
            axs[axi, ex].axes.get_yaxis().set_ticks([])
        axs[axi, 0].set_ylabel(lab)

    # plt.show()
    # plt.savefig('output/vaegan_output/image_at_epoch_{:04d}.png'.format(epoch))
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, 'image_at_epoch_{:04d}.png'.format(epoch)))


def plot_losses(losses, epoch, path):
    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(16, 4))
    axs[0].plot(losses.latent_loss.values, label='latent_loss')
    axs[1].plot(losses.discrim_layer_recon_loss.values, label='discrim_layer_recon_loss')
    axs[2].plot(losses.disc_real_loss.values, label='disc_real_loss')
    axs[2].plot(losses.disc_fake_loss.values, label='disc_fake_loss')
    axs[2].plot(losses.gen_fake_loss.values, label='gen_fake_loss')
    axs[3].plot(losses.d_prop.values, label='d_prop')

    for ax in axs.flatten():
        ax.legend()
    # plt.show()
    # plt.savefig('output/vaegan_output/loss_at_epoch_{:04d}.png'.format(epoch))
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, 'loss_at_epoch_{:04d}.png'.format(epoch)))


def vaegan_train(train_dataset, test_dataset, pretrained_path=None):
    print('start')
    N_TRAIN_BATCHES = int(3200 / config.BATCH_SIZE)
    N_TEST_BATCHES = int(3200 / config.BATCH_SIZE)
    example_data = next(iter(train_dataset))

    pretrained_model = VAEGAN(
        lr_base_gen=1e-3,  #
        lr_base_disc=1e-4,  # the discriminator's job is easier than the generators so make the learning rate lower
        latent_loss_div=1,
        # this variable will depend on your dataset - choose a number that will bring your latent loss to ~1-10
        sig_mult=10,  # how binary the discriminator's learning rate is shifted (we squash it with a sigmoid)
        recon_loss_div=.001,
        # this variable will depend on your dataset - choose a number that will bring your latent loss to ~1-10
    )

    # a pandas dataframe to save the loss information to
    losses = pd.DataFrame(columns=[
        'd_prop',
        'latent_loss',
        'discrim_layer_recon_loss',
        'gen_fake_loss',
        'disc_fake_loss',
        'disc_real_loss',
    ])
    pretrained_checkpoint_dir = './vaegan_training_checkpoints'
    pretrained_checkpoint_prefix = os.path.join(pretrained_checkpoint_dir, "ckpt")
    pretrained_checkpoint = tf.train.Checkpoint(enc_optimizer=pretrained_model.enc_optimizer,
                                                dec_optimizer=pretrained_model.dec_optimizer,
                                                disc_optimizer=pretrained_model.disc_optimizer,
                                                enc=pretrained_model.enc,
                                                dec=pretrained_model.dec,
                                                disc=pretrained_model.disc)

    model = VAEGAN(
        lr_base_gen=1e-4,  #
        lr_base_disc=1e-5,  # the discriminator's job is easier than the generators so make the learning rate lower
        latent_loss_div=1,
        # this variable will depend on your dataset - choose a number that will bring your latent loss to ~1-10
        sig_mult=10,  # how binary the discriminator's learning rate is shifted (we squash it with a sigmoid)
        recon_loss_div=.001,
        # this variable will depend on your dataset - choose a number that will bring your latent loss to ~1-10
    )
    if pretrained_path is not None:
        status = pretrained_checkpoint.restore(pretrained_path)
        print(status)
        print("Restore checkpoint done. ")
        checkpoint_dir = './vaegan_finetune_training_checkpoints'
        output_path = './output/vaegan_finetune_output/'
        model.disc = pretrained_checkpoint.disc
        model.enc = pretrained_checkpoint.enc
        model.dec = pretrained_checkpoint.dec

    else:
        checkpoint_dir = './vaegan_training_checkpoints'
        output_path = './output/vaegan_output/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(enc_optimizer=model.enc_optimizer,
                                     dec_optimizer=model.dec_optimizer,
                                     disc_optimizer=model.disc_optimizer,
                                     enc=model.enc,
                                     dec=model.dec,
                                     disc=model.disc)

    n_epochs = 200
    print('checkpoint construction completed. ')
    for epoch in range(n_epochs):
        # train
        print('train: ', epoch)
        start_time = time.time()
        for train_x in tqdm(train_dataset):
            model.train(train_x)
        # test on holdout
        loss = []
        for test_x in test_dataset:
            loss.append(model.compute_loss(test_x))
        losses.loc[len(losses)] = np.mean(loss, axis=0)
        print('test: ', epoch)
        end_time = time.time()
        # plot results
        print(
            "Epoch: {} loss: {} elapsed time: {}".format(epoch, np.sum(np.mean(loss, axis=0)), end_time - start_time)
        )
        checkpoint.save(file_prefix=checkpoint_prefix + "_epoch_%d" % (epoch + 1))
        plot_reconstruction(model, example_data, epoch=epoch, path=output_path)
        plot_losses(losses, epoch, output_path)


# classifier
classifier = make_classifier()
# classifier.compile(optimizer='adam',
#                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                    metrics=['accuracy'])

classifier_opt = tf.keras.optimizers.Adam(config.LR)


@tf.function
def classifier_train_step(x, frozen_generator, frozen_discriminator,
                          frozen_inv_gen, y_true):
    with tf.GradientTape() as tape:
        recon_img = frozen_generator(frozen_inv_gen(x))
        dis_recon = frozen_discriminator(recon_img)
        dis_real = frozen_discriminator(x)

        diff_recon = tf.subtract(x, recon_img)
        diff_dis = tf.subtract(dis_real, dis_recon)

        y_pred = classifier([diff_recon, diff_dis, dis_real])
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)
    gradients = tape.gradient(loss, classifier.trainable_variables)
    classifier_opt.apply_gradients(zip(gradients, classifier.trainable_variables))
    return loss


def classifier_train(train_dataset, test_dataset, pretriained_path=None):
    gan_checkpoint_dir = './gan_training_checkpoints'
    gan_checkpoint = tf.train.Checkpoint(
        optimizer=generator_opt,
        discriminator_optimizer=discriminator_opt,
        generator=generator,
        discriminator=discriminator,
    )

    latest = tf.train.latest_checkpoint(gan_checkpoint_dir)
    gan_checkpoint.restore(latest)
    frozen_generator = gan_checkpoint.generator
    aae = AAE(gan_checkpoint)
    aae_checkpoint = tf.train.Checkpoint(AE_optimizer=aae.optimizer,
                                         AE=aae)
    aae_latest = tf.train.latest_checkpoint('./aae_training_checkpoints')
    aae_checkpoint.restore(aae_latest)

    frozen_invG = aae_checkpoint.AE.inference_net
    frozen_discriminator = aae_checkpoint.AE.discriminator
    frozen_discriminator.trainable = False
    frozen_invG.trainable = False
    frozen_generator.trainable = False
    for epoch in range(config.EPOCHS):
        print("epoch: ", epoch)
        total_loss = 0
        for imgs, labels in tqdm(train_dataset):
            loss = classifier_train_step(imgs, frozen_generator, frozen_discriminator, frozen_invG,
                                         labels)
            total_loss += loss
        print(total_loss.numpy())

        # test
        acc = []
        for x, labels in tqdm(test_dataset):
            recon_img = frozen_generator(frozen_invG(x))
            dis_recon = frozen_discriminator(recon_img)
            dis_real = frozen_discriminator(x)

            diff_recon = tf.subtract(x, recon_img)
            diff_dis = tf.subtract(dis_real, dis_recon)

            y_pred = classifier([diff_recon, diff_dis, dis_real])

            m = tf.keras.metrics.BinaryAccuracy()
            m.update_state(labels, y_pred)
            acc.append(m.result().numpy())
        print("test: ", np.mean(acc))


def main(_):
    flags_dict = FLAGS.flag_values_dict()
    train_mode = flags_dict['train_mode']
    print(train_mode)

    if train_mode is not TrainMode.FL:
        train_dataset = build_dataset(config.TRAIN_DATA_PATH, positive_only=True).shuffle(3200).batch(config.BATCH_SIZE)
        test_dataset = build_dataset(config.TEST_DATA_PATH, positive_only=True).shuffle(3200).batch(config.BATCH_SIZE)
    else:
        train_dataset = build_dataset(
            config.FL_TRAIN_DATA_PATH, positive_only=False).shuffle(3200).batch(config.BATCH_SIZE)
        test_dataset = build_dataset(
            config.FL_TEST_DATA_PATH, positive_only=False).shuffle(3200).batch(config.BATCH_SIZE)


    if train_mode in [TrainMode.GAN, TrainMode.ALL]:
        print("************************* Training GAN ***************************")
        gan_train(train_dataset, config.EPOCHS)
    elif train_mode in [TrainMode.AE, TrainMode.AAE, TrainMode.ALL]:
        print("************************* Training AE ***************************")
        ae_train(train_dataset, test_dataset, config.EPOCHS, train_mode)

    elif train_mode == TrainMode.VAEGAN:
        print("************************* Training VAE-GAN ***************************")
        # vaegan_train(train_dataset, test_dataset, pretrained_path='/data2/huangps/data_freakie/gx/code/elevator_GAN_FH/vaegan_training_checkpoints/ckpt_epoch_60-60')
        vaegan_train(train_dataset, test_dataset)

    elif train_mode == TrainMode.FL:
        print("************************* Training VAE-GAN ***************************")
        classifier_train(train_dataset, test_dataset)


if __name__ == '__main__':
    app.run(main)
