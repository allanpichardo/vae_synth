import os
from glob import glob

import kapre
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SoundSequence(tf.keras.utils.Sequence):

    def __init__(self, music_path, sr=44100, duration=1.0, batch_size=32, shuffle=True):
        """
        Create a data generator that reads wav files from a directory
        :param music_path:
        :param duration: duration of sound clips
        :param batch_size:
        :param shuffle:
        """
        self.sr = sr
        self.duration = duration
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.stft_func = kapre.composed.get_stft_mag_phase((int(sr * duration), 1))

        self.wav_paths = glob(os.path.join(music_path, '*', '*'))
        self.labels = []
        for path in self.wav_paths:
            real_label = os.path.basename(os.path.dirname(path))
            self.labels.append(real_label)

        self.on_epoch_end()

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        X = []
        Y = []

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            wav, rate = librosa.load(path, sr=self.sr, duration=self.duration, res_type='kaiser_fast')
            wav = tf.convert_to_tensor(wav)
            wav = tf.expand_dims(wav, 1)
            wav = self.pad_up_to(wav, [rate * int(self.duration), 1], 0)
            img = self.stft_func(tf.stack([wav]))
            img = tf.image.per_image_standardization(img)
            padded = tf.image.resize_with_crop_or_pad(img, 256, 1025)
            X.append(tf.squeeze(padded, 0))
            Y.append(tf.convert_to_tensor(label))

        X = tf.stack(X)
        Y = tf.stack(Y)

        return X, Y

    def __len__(self):
        return int(np.ceil(len(self.wav_paths) / float(self.batch_size)))

    def pad_up_to(self, t, max_in_dims, constant_values):
        s = tf.shape(t)
        paddings = [[0, m - s[i]] for (i, m) in enumerate(max_in_dims)]
        return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):

    def call(self, inputs, training=None, mask=None):
        u, v, z = self.encoder(inputs)
        y_pred = self.decoder(z)
        return y_pred

    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def pad_up_to(self, t, max_in_dims, constant_values):
        s = tf.shape(t)
        paddings = [[0, m - s[i]] for (i, m) in enumerate(max_in_dims)]
        return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            mag_true, _ = tf.split(data, 2, 3)
            mag_pred, _ = tf.split(reconstruction, 2, 3)
            spectral_convergence_loss = tf.sqrt(
                tf.divide(
                    tf.reduce_sum(tf.square(mag_true - mag_pred)),
                    tf.reduce_sum(tf.square(mag_true))
                )
            )

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = spectral_convergence_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "spectral_conv_loss": spectral_convergence_loss,
            "kl_loss": kl_loss,
        }


def img_to_complex(x):
    m, p = tf.split(x, 2, axis=3)
    return tf.complex(m, p)


def get_models_cnn(latent_dim=16, input_shape=(256, 1025, 2)):
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, padding="same")(encoder_inputs)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.Conv2D(1, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(1024, activation="relu")(latent_inputs)
    x = layers.Reshape((16, 64, 1))(x)
    x = layers.Conv2DTranspose(1, 3, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2DTranspose(64, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2DTranspose(64, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2DTranspose(64, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D()(x)
    x = layers.ZeroPadding2D(padding=[(0, 0), (1, 0)])(x)
    x = layers.Conv2DTranspose(64, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    decoder_outputs = layers.Conv2DTranspose(2, 3, activation=None, padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = VAE(encoder, decoder)

    return encoder, decoder, vae


if __name__ == '__main__':
    path = '/Users/allanpichardo/Downloads/Legowelt QuasiMIDI SIRIUS Sample Pack'
    sr = 22050
    duration = 3.0
    batch_size = 32

    sequence = SoundSequence(path, sr=sr, duration=duration, batch_size=32)

    encoder, decoder, autoencoder = get_models_cnn(latent_dim=8)
    encoder.summary()
    decoder.summary()

    # autoencoder.compile(optimizer=keras.optimizers.Adam())
    # autoencoder.fit(sequence, epochs=10)

    # Y = autoencoder.predict_on_batch([y])

    # for y in Y:
    #     wav = tf.audio.encode_wav(y, sr)
    #     tf.io.write_file('output.wav', wav)
