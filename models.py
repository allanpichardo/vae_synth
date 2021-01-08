import os
from glob import glob

import kapre
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from tensorboard.plugins import projector
from griffin_lim import GriffinLim, STFTNormalize


class SpectrogramCallback(tf.keras.callbacks.Callback):

    def __init__(self, soundsequence, sr=44100, logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")):
        super().__init__()
        self.soundequence = soundsequence
        self.logdir = logdir
        self.sr = sr

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.soundequence.__getitem__(0)

        spec_x = self.model.stft(x)
        embedding = self.model.encoder(spec_x)
        spec_y = self.model.decoder(embedding)
        audio_y = GriffinLim()(spec_y)

        checkpoint = tf.train.Checkpoint(embedding=tf.Variable(embedding))
        checkpoint.save(os.path.join(self.logdir, 'embedding.ckpt'))
        config = projector.ProjectorConfig()
        embs = config.embeddings.add()
        embs.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        projector.visualize_embeddings(self.logdir, config)

        file_writer = tf.summary.create_file_writer(self.logdir)

        with file_writer.as_default():
            tf.summary.audio("Sample Input", x, self.sr, step=epoch, max_outputs=5, description="Audio sample input")
            tf.summary.image("STFT Input", spec_x, step=epoch, max_outputs=5, description="Spectrogram input")
            tf.summary.image("STFT Reconstruction", spec_y, step=epoch, max_outputs=5, description="Spectrogram output")
            tf.summary.audio("Sample Reconstruction", audio_y, self.sr, step=epoch, max_outputs=5,
                             description="Synthesized audio")


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
        return int(np.floor(len(self.wav_paths) / float(self.batch_size)))

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
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            stft_out = self.stft(data)
            z_mean, z_log_var, z = self.encoder(stft_out)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.sqrt(
                tf.divide(
                    tf.reduce_sum(tf.square(stft_out - reconstruction)),
                    tf.reduce_sum(tf.square(stft_out))
                )
            )
            #
            # kl = 0.5 * tf.reduce_sum(tf.exp(z_log_var) + tf.square(z_mean) - 1. - z_log_var, axis=1)
            #
            # total_loss = spectral_convergence_loss + kl

            # reconstruction_loss = tf.reduce_mean(
            #     tf.reduce_sum(
            #         keras.losses.mae(stft_out, reconstruction), axis=(1, 2)
            #     )
            # )
            coefficient = 0.0001
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) * coefficient
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def get_synth_model(decoder, input_shape=(8,)):
    inputs = keras.Input(shape=input_shape)
    x = decoder(inputs)
    x = GriffinLim()(x)
    return keras.Model(inputs, x, name="synth")


def get_model(latent_dim=8, sr=44100, duration=3.0):
    input_shape = (int(sr * duration), 1)
    encoder_inputs = keras.Input(shape=input_shape)
    x = kapre.STFT(input_shape=input_shape, n_fft=1024)(encoder_inputs)
    x = kapre.Magnitude()(x)
    x = kapre.MagnitudeToDecibel()(x)
    x = STFTNormalize()(x)
    stft_out = layers.Lambda(lambda m: tf.image.resize_with_crop_or_pad(m, 513, 513))(x)
    stft_model = keras.Model(encoder_inputs, stft_out, name='stft')

    img_inputs = keras.Input(shape=(513, 513, 1))
    x = layers.TimeDistributed(layers.Conv1D(32, 3, padding="same"))(img_inputs)
    x = layers.LeakyReLU()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.TimeDistributed(layers.Conv1D(32, 3, padding="same"))(x)
    x = layers.LeakyReLU()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.TimeDistributed(layers.Conv1D(64, 3, padding="same"))(x)
    x = layers.LeakyReLU()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.TimeDistributed(layers.Conv1D(64, 3, padding="same"))(x)
    x = layers.TimeDistributed(layers.Conv1D(1, 3, padding="same"))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean", activation=None)(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var", activation=None)(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(img_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(64 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((64, 64, 1))(x)
    x = layers.TimeDistributed(layers.Conv1DTranspose(64, 3, padding="same"))(x)
    x = layers.LeakyReLU()(x)
    x = layers.UpSampling2D()(x)
    x = layers.TimeDistributed(layers.Conv1DTranspose(64, 3, padding="same"))(x)
    x = layers.LeakyReLU()(x)
    x = layers.UpSampling2D()(x)
    x = layers.TimeDistributed(layers.Conv1DTranspose(32, 3, padding="same"))(x)
    x = layers.LeakyReLU()(x)
    x = layers.UpSampling2D()(x)
    x = layers.ZeroPadding2D(padding=[(0, 1), (0, 1)])(x)
    x = layers.TimeDistributed(layers.Conv1DTranspose(32, 3, padding="same"))(x)
    x = layers.LeakyReLU()(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation=None, padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = VAE(stft_model, encoder, decoder)

    return vae


def pad_up_to(t, max_in_dims, constant_values):
    s = tf.shape(t)
    paddings = [[0, m - s[i]] for (i, m) in enumerate(max_in_dims)]
    return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)


if __name__ == '__main__':
    logdir = os.path.join(os.path.dirname(__file__), 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))

    stft_model_path = os.path.join(os.path.dirname(__file__), 'models', 'stft_mod_v{}'.format(1))
    enc_model_path = os.path.join(os.path.dirname(__file__), 'models', 'enc_mod_v{}'.format(1))
    dec_model_path = os.path.join(os.path.dirname(__file__), 'models', 'dec_mod_v{}'.format(1))

    path = os.path.join(os.path.dirname(__file__), 'samples')
    sr = 44100
    duration = 3.0
    batch_size = 4
    latent_dim = 10

    sequence = SoundSequence(path, sr=sr, duration=duration, batch_size=batch_size)

    autoencoder = None
    if os.path.exists(stft_model_path) and os.path.exists(enc_model_path) and os.path.exists(dec_model_path):
        autoencoder = VAE(
            tf.keras.models.load_model(stft_model_path, compile=False),
            tf.keras.models.load_model(enc_model_path, compile=False),
            tf.keras.models.load_model(dec_model_path, compile=False)
        )
    else:
        autoencoder = get_model(latent_dim=latent_dim, sr=sr, duration=duration)

    autoencoder.stft.summary()
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    # wav, rate = librosa.load('/Users/allanpichardo/PycharmProjects/audio-generation-autoencoder/FX-Robotio.wav', sr=sr,
    #                          duration=duration)
    # wav = tf.convert_to_tensor(wav)
    # wav = tf.expand_dims(wav, axis=1)
    # wav = pad_up_to(wav, [int(duration * sr), 1], 0)
    # wav = tf.expand_dims(wav, axis=0)
    # stft = autoencoder.stft(wav)
    # repro = GriffinLim()(stft)
    # repro = repro[0]
    # repro = librosa.util.normalize(repro)
    # repro = tf.audio.encode_wav(repro, 44100)
    # tf.io.write_file('reproduction.wav', repro)

    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005))
    autoencoder.fit(sequence, epochs=15, callbacks=[
        SpectrogramCallback(sequence, sr=sr),
        tf.keras.callbacks.TensorBoard(log_dir=logdir)
    ])

    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'models')):
        os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)

    autoencoder.stft.save(stft_model_path, save_format='tf', include_optimizer=False)
    autoencoder.encoder.save(enc_model_path, save_format='tf', include_optimizer=False)
    autoencoder.decoder.save(dec_model_path, save_format='tf', include_optimizer=False)

    synth = get_synth_model(autoencoder.decoder)
    synth.summary()
    #
    random = tf.random.normal([5, latent_dim])
    wavs = synth.predict_on_batch(random)

    i = 0
    for wav in wavs:
        wav = librosa.util.normalize(wav)
        wav = tf.audio.encode_wav(wav, sr)
        tf.io.write_file('output-{}.wav'.format(i), wav)
        i = i + 1
