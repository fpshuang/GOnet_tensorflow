# Created by pengsheng.huang at 8/10/2020

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Conv2DTranspose, BatchNormalization, ReLU, Reshape, Flatten, ELU
import numpy as np
import pandas as pd
import config
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import tensorflow_probability as tfp
ds = tfp.distributions


def make_classifier():
    l_img_input = tf.keras.Input(shape=(128, 128, 1), name='image_diff_input')
    l_img = tf.keras.layers.Lambda(tf.math.abs)(l_img_input)
    l_img = Flatten()(l_img)
    l_img = Dense(1, input_shape=(128 * 128, ),  activation='linear')(l_img)

    l_dis_input = tf.keras.Input(shape=(8, 8, 512), name='diff_dis_input')
    l_dis = tf.keras.layers.Lambda(tf.math.abs)(l_dis_input)
    l_dis = Flatten()(l_dis)
    l_dis = Dense(1, input_shape=(8 * 8 * 512, ), activation='linear')(l_dis)

    l_fdis_input = tf.keras.Input(shape=(8, 8, 512), name='dis_real_input')
    l_fdis = tf.keras.layers.Lambda(tf.math.abs)(l_fdis_input)
    l_fdis = Flatten()(l_fdis)
    l_fdis = Dense(1, input_shape=(8 * 8 * 512, ), activation='linear')(l_fdis)

    fl = tf.keras.layers.concatenate([l_img, l_dis, l_fdis])
    fl = Dense(1, input_shape=(3, ), activation='sigmoid')(fl)

    return Model(inputs=[l_img_input, l_dis_input, l_fdis_input],
                 outputs=fl)


class AAE(Model):
    def __init__(self, checkpoint):
        super(AAE, self).__init__()
        self.latent_dim = config.NZ
        self.inference_net = invG().build()
        self.generator_net = checkpoint.generator
        ori_discriminator = checkpoint.discriminator
        self.discriminator = Model(inputs=ori_discriminator.inputs,
                                   outputs=[ori_discriminator.get_layer('conv2d_3').output])
        self.optimizer = tf.keras.optimizers.Adam(config.LR * 10)

    @tf.function
    def decode(self, latent):
        return self.generator_net(latent)

    @tf.function
    def encode(self, x):
        return self.inference_net(x)

    @tf.function
    def get_disc_last_conv(self, x):
        return self.discriminator.get_layer('conv2d_3').output

    @tf.function
    def compute_loss(self, x):
        latent = self.encode(x)
        _x = self.decode(latent)
        dis_ori = self.discriminator(x)
        dis_gen = self.discriminator(_x)
        ld = tf.reduce_mean(tf.square(dis_ori - dis_gen))
        lr = tf.reduce_mean(tf.square(x - _x))
        return 100 * config.LAMBDA * lr + (1 - config.LAMBDA) * ld

    @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)

    @tf.function
    def train(self, train_x):
        gradients = self.compute_gradients(train_x)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def represent(self, x):
        return self.decode(self.encode(x))


class AE(Model):
    def __init__(self, generator_net):
        super(AE, self).__init__()
        self.latent_dim = config.NZ
        self.inference_net = invG().build()
        self.generator_net = generator_net
        initial_learning_rate = 1e-2
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(lr_schedule)

    @tf.function
    def decode(self, latent):
        return self.generator_net(latent)

    @tf.function
    def encode(self, x):
        return self.inference_net(x)

    @tf.function
    def compute_loss(self, x):
        latent = self.encode(x)
        _x = self.decode(latent)
        return tf.reduce_mean(tf.square(x - _x))

    @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)

    @tf.function
    def train(self, train_x):
        gradients = self.compute_gradients(train_x)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def represent(self, x):
        return self.decode(self.encode(x))


class Generator:
    def __init__(self, name=None, **kwargs):
        self.l0z = Dense(8 * 8 * 512, use_bias=False, input_shape=(config.NZ, ))
        self.dc1 = Conv2DTranspose(256, 4, strides=(2, 2), padding='same')
        self.dc2 = Conv2DTranspose(128, 4, strides=(2, 2), padding='same')
        self.dc3 = Conv2DTranspose(64, 4, strides=(2, 2), padding='same')
        self.dc4 = Conv2DTranspose(1, 4, strides=(2, 2), padding='same')
        self.bn0 = BatchNormalization()
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.bn4 = BatchNormalization()
        self.ac1 = ReLU()
        self.ac2 = ReLU()
        self.ac3 = ReLU()
        self.ac0 = ReLU()
        self.reshape = Reshape((8, 8, 512))

    def build(self):
        inputs = tf.keras.Input(shape=(100,), name='input')
        x = self.bn0(self.l0z(inputs))
        x = self.ac0(x)
        x = self.reshape(x)

        x = self.dc1(x)
        x = self.bn1(x)
        x = self.ac1(x)

        x = self.dc2(x)
        x = self.bn2(x)
        x = self.ac2(x)

        x = self.dc3(x)
        x = self.bn3(x)
        x = self.ac3(x)
        x = self.dc4(x)

        return Model(inputs=[inputs], outputs=[x])


class invG:
    def __init__(self, multiplier=1, name=None, **kwargs):
        self.c0 = Conv2D(64, 4, strides=(2, 2), padding='same')
        self.c1 = Conv2D(128, 4, strides=(2, 2), padding='same')
        self.c2 = Conv2D(256, 4, strides=(2, 2), padding='same')
        self.c3 = Conv2D(512, 4, strides=(2, 2), padding='same')
        self.l4l = Dense(config.NZ * multiplier, input_shape=(8*8*512, ), activation='linear')
        self.bn0 = BatchNormalization()
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.ac1 = ReLU()
        self.ac2 = ReLU()
        self.ac3 = ReLU()
        self.ac0 = ReLU()
        self.reshape = Flatten()

    def build(self):
        inputs = tf.keras.Input(shape=(128, 128, 1), name='input')
        x = self.c0(inputs)
        x = self.ac0(x)

        x = self.c1(x)
        x = self.bn1(x)
        x = self.ac1(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = self.ac2(x)

        x = self.c3(x)
        x = self.bn3(x)
        x = self.ac3(x)
        x = self.reshape(x)

        x = self.l4l(x)
        return Model(inputs=[inputs], outputs=[x])


class Discriminator:
    def __init__(self, name=None, **kwargs):
        self.c0 = Conv2D(64, 4, strides=(2, 2), padding='same')
        self.c1 = Conv2D(128, 4, strides=(2, 2), padding='same')
        self.c2 = Conv2D(256, 4, strides=(2, 2), padding='same')
        self.c3 = Conv2D(512, 4, strides=(2, 2), padding='same')
        self.l4l = Dense(1, input_shape=(8*8*512, ), activation='linear')
        self.bn0 = BatchNormalization()
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.elu0 = ELU()
        self.elu1 = ELU()
        self.elu2 = ELU()
        self.elu3 = ELU()
        self.reshape = Flatten()

    def build(self, return_model=True):
        inputs = tf.keras.Input(shape=(128, 128, 1), name='input')
        x = self.c0(inputs)
        x = self.elu0(x)

        x = self.c1(x)
        x = self.bn1(x)
        x = self.elu1(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = self.elu2(x)

        x = self.c3(x)
        last_layer = x
        x = self.bn3(x)
        x = self.elu3(x)

        x = self.reshape(x)
        x = self.l4l(x)
        if return_model:
            return Model(inputs=[inputs], outputs=[x])
        return inputs, last_layer, x


class VAEGAN(tf.keras.Model):
    """a VAEGAN class for tensorflow

    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(VAEGAN, self).__init__()
        self.__dict__.update(kwargs)

        self.enc = invG(multiplier=2).build()
        self.dec = Generator().build()
        inputs, disc_l, outputs = Discriminator().build(return_model=False)
        self.disc = tf.keras.Model(inputs=[inputs], outputs=[outputs, disc_l])

        self.enc_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.5)
        self.dec_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(self.get_lr_d, beta_1=0.5)

    def encode(self, x):
        mu, sigma = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return mu, sigma

    def dist_encode(self, x):
        mu, sigma = self.encode(x)
        return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    def get_lr_d(self):
        return self.lr_base_disc * self.D_prop

    def decode(self, latent):
        # print("latent", latent)
        return self.dec(latent)

    def discriminate(self, x):
        return self.disc(x)

    def reconstruct(self, x):
        mean, _ = self.encode(x)
        return self.decode(mean)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    # @tf.function
    def compute_loss(self, x):
        # pass through network
        q_z = self.dist_encode(x)
        latent = q_z.sample()
        p_z = ds.MultivariateNormalDiag(
            loc=[0.0] * latent.shape[-1], scale_diag=[1.0] * latent.shape[-1]
        )
        # print('z: ', latent)
        xg = self.decode(latent)
        z_samp = tf.random.normal([x.shape[0], latent.shape[-1]])
        xg_samp = self.decode(z_samp)
        d_xg, ld_xg = self.discriminate(xg)
        d_x, ld_x = self.discriminate(x)
        d_xg_samp, ld_xg_samp = self.discriminate(xg_samp)

        # GAN losses
        disc_real_loss = gan_loss(logits=d_x, is_real=True)
        disc_fake_loss = gan_loss(logits=d_xg_samp, is_real=False)
        gen_fake_loss = gan_loss(logits=d_xg_samp, is_real=True)

        discrim_layer_recon_loss = (
                tf.reduce_mean(tf.reduce_mean(tf.math.square(ld_x - ld_xg), axis=0))
                / self.recon_loss_div
        )

        self.D_prop = sigmoid(
            disc_fake_loss - gen_fake_loss, shift=0.0, mult=self.sig_mult
        )

        kl_div = ds.kl_divergence(q_z, p_z)
        latent_loss = tf.reduce_mean(tf.maximum(kl_div, 0)) / self.latent_loss_div

        return (
            self.D_prop,
            latent_loss,
            discrim_layer_recon_loss,
            gen_fake_loss,
            disc_fake_loss,
            disc_real_loss,
        )

    # @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:
            (
                _,
                latent_loss,
                discrim_layer_recon_loss,
                gen_fake_loss,
                disc_fake_loss,
                disc_real_loss,
            ) = self.compute_loss(x)

            enc_loss = latent_loss + discrim_layer_recon_loss
            dec_loss = gen_fake_loss + discrim_layer_recon_loss
            disc_loss = disc_fake_loss + disc_real_loss

        enc_gradients = enc_tape.gradient(enc_loss, self.enc.trainable_variables)
        dec_gradients = dec_tape.gradient(dec_loss, self.dec.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        return enc_gradients, dec_gradients, disc_gradients

    @tf.function
    def apply_gradients(self, enc_gradients, dec_gradients, disc_gradients):
        self.enc_optimizer.apply_gradients(
            zip(enc_gradients, self.enc.trainable_variables)
        )
        self.dec_optimizer.apply_gradients(
            zip(dec_gradients, self.dec.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.disc.trainable_variables)
        )

    def train(self, x):
        enc_gradients, dec_gradients, disc_gradients = self.compute_gradients(x)
        self.apply_gradients(enc_gradients, dec_gradients, disc_gradients)


def gan_loss(logits, is_real=True):
    """Computes standard gan loss between logits and labels

        Arguments:
            logits {[type]} -- output of discriminator

        Keyword Arguments:
            isreal {bool} -- whether labels should be 0 (fake) or 1 (real) (default: {True})
        """
    if is_real:
        labels = tf.ones_like(logits)
    else:
        labels = tf.zeros_like(logits)

    return tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits
    )


def sigmoid(x, shift=0.0, mult=20):
    """ squashes a value with a sigmoid
    """
    return tf.constant(1.0) / (
            tf.constant(1.0) + tf.exp(-tf.constant(1.0) * (x * mult))
    )


if __name__ == '__main__':
    # test generator.
    generator = Generator().build()
    tf.keras.utils.plot_model(
        generator, to_file='gen.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)

    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    plt.show()

    # test invGenerator.
    invGenerator = invG().build()
    tf.keras.utils.plot_model(
        invGenerator, to_file='invG.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )

    z = invGenerator(generated_image)

    print("z: ", z)

    generated_image = generator(z, training=False)

    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    plt.show()
    print("inv-generator out shape: ", z.shape)
    print("noise shape: ", noise.shape)

    # test discriminator.

    discriminator = Discriminator()
    discriminator = discriminator.build()
    tf.keras.utils.plot_model(
        discriminator, to_file='disc.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )
    print(generated_image.shape)
    dis_out = discriminator(generated_image, training=False)

    print("dis out: ", dis_out)

    # test ae
    generator.trainable = False
    model = AE(generator)
    print(model.generator_net.trainable)
    print(len(model.trainable_variables))
    print(len(model.non_trainable_variables))

    generated_image = model.represent(generated_image)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    plt.show()

    # test vae gan.
    vaegan = VAEGAN(
        lr_base_gen=1e-3,  #
        lr_base_disc=1e-4,  # the discriminator's job is easier than the generators so make the learning rate lower
        latent_loss_div=1,
        # this variable will depend on your dataset - choose a number that will bring your latent loss to ~1-10
        sig_mult=10,  # how binary the discriminator's learning rate is shifted (we squash it with a sigmoid)
        recon_loss_div=.001,
        # this variable will depend on your dataset - choose a number that will bring your latent loss to ~1-10
    )

    # test classifier.
    classifier = make_classifier()
    print(classifier.summary())
