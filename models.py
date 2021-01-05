import os
from glob import glob

import kapre
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SoundSequence(tf.keras.utils.Sequence):

    def __init__(self, music_path, n_fft=2048, sr=44100, duration=2.0, batch_size=32, shuffle=True):
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
            X.append(wav)
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
        spec = self.stft(inputs)
        u, v, z = self.encoder(spec)
        y_pred = self.decoder(z)
        return y_pred

    def __init__(self, stft, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.stft = stft
        self.encoder = encoder
        self.decoder = decoder

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            stft_out = self.stft(data)
            z_mean, z_log_var, z = self.encoder(stft_out)
            reconstruction = self.decoder(z)

            spectral_convergence_loss = tf.sqrt(
                tf.divide(
                    tf.reduce_sum(tf.square(stft_out - reconstruction)),
                    tf.reduce_sum(tf.square(stft_out))
                )
            )

            kl = 0.5 * tf.reduce_sum(tf.exp(z_log_var) + tf.square(z_mean) - 1. - z_log_var, axis=1)

            total_loss = spectral_convergence_loss + kl

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "spectral_conv_loss": spectral_convergence_loss,
            "kl_loss": kl,
        }


def img_to_complex(x):
    m, p = tf.split(x, 2, axis=3)
    return tf.complex(m, p)


def get_synth_model(decoder, input_shape=(8,)):
    inputs = keras.Input(shape=input_shape)
    x = decoder(inputs)
    x = stft_to_wav_model()(x)
    return keras.Model(inputs, x, name="synth")


def stft_to_wav_model(input_shape=(513, 513, 2)):
    inputs = keras.Input(shape=input_shape)
    x = tf.multiply(tf.constant(2.0), inputs)
    x = tf.subtract(x, 1)
    m, p = tf.split(x, 2, 3)
    x = tf.complex(m, p)
    x = kapre.InverseSTFT(n_fft=1024)(x)
    return keras.Model(inputs, x, name="InverseStft")


def get_model(latent_dim=8, sr=44100, duration=3.0):
    input_shape = (int(sr * duration), 1)
    encoder_inputs = keras.Input(shape=input_shape)
    x = kapre.composed.get_stft_mag_phase(input_shape=input_shape, n_fft=1024)(encoder_inputs)
    stft_out = tf.image.resize_with_crop_or_pad(x, 513, 513)
    stft_model = keras.Model(encoder_inputs, stft_out, name='stft')

    img_inputs = keras.Input(shape=(513, 513, 2))
    x = layers.Conv2D(32, 3, padding="same")(img_inputs)
    x = layers.LeakyReLU()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean", activation="linear")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var", activation="linear")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(img_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(256 * 256, activation="relu")(latent_inputs)
    x = layers.Reshape((256, 256, 1))(x)
    x = layers.Conv2DTranspose(64, 3, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2DTranspose(32, 3, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.ZeroPadding2D(padding=[(0, 1), (0, 1)])(x)
    decoder_outputs = layers.Conv2DTranspose(2, 3, activation="linear", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = VAE(stft_model, encoder, decoder)

    return vae


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), 'samples')
    sr = 44100
    duration = 3.0
    batch_size = 4

    sequence = SoundSequence(path, sr=sr, duration=duration, batch_size=batch_size)

    autoencoder = get_model(latent_dim=8, sr=sr, duration=duration)
    autoencoder.stft.summary()
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    autoencoder.compile(optimizer=keras.optimizers.Adam())
    autoencoder.fit(sequence, epochs=1)

    synth = get_synth_model(autoencoder.decoder)
    synth.summary()

    random = tf.constant(np.random.random(8), shape=(1, 8,))
    wav = synth.predict_on_batch(random)
    wav = librosa.util.normalize(wav[0])

    wav = tf.audio.encode_wav(wav, sr)
    tf.io.write_file('output.wav', wav)
