# Created by pengsheng.huang at 8/19/2020

from model import AE, Discriminator, Generator, invG
import tensorflow as tf
import config
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # seed = tf.random.normal([4, config.NZ])
    generator = Generator().build()
    discriminator = Discriminator().build()
    inv_generator = invG().build()

    generator_opt = tf.keras.optimizers.Adam(config.LR)
    discriminator_opt = tf.keras.optimizers.Adam(config.LR)

    # gan_checkpoint_dir = './gan_training_checkpoints'
    # gan_checkpoint = tf.train.Checkpoint(
    #     optimizer=generator_opt,
    #     discriminator_optimizer=discriminator_opt,
    #     generator=generator,
    #     discriminator=discriminator,
    # )
    #
    # latest = tf.train.latest_checkpoint(gan_checkpoint_dir)
    # gan_checkpoint.restore(latest)
    #
    # frozen_generator = gan_checkpoint.generator
    # frozen_discriminator = gan_checkpoint.discriminator
    # frozen_generator.trainable = False

    # load Auto-decoder.
    model = AE(generator)
    ae_checkpoint_dir = './ae_training_checkpoints'
    ae_checkpoint = tf.train.Checkpoint(
        checkpoint=tf.train.Checkpoint(optimizer=model.optimizer,
                                       model=model)
    )
    latest = tf.train.latest_checkpoint(ae_checkpoint_dir)
    status = ae_checkpoint.restore(latest)
    print(status)
    print(latest)
    print(dir(ae_checkpoint))
    encoder = ae_checkpoint.model.inference_net

    test_img = cv2.imread('/data2/huangps/data_freakie/gx/data/gonet_original/data_test_annotation/negative_L/img_negative_L_99.jpg')
    latent = encoder(test_img)
    print(latent)
