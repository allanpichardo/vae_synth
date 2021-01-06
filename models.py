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
            # speed = random.uniform(0.25, 2.0)
            wav, rate = librosa.load(path, sr=self.sr, duration=self.duration, res_type='kaiser_fast')
            # wav = librosa.effects.time_stretch(wav, speed)
            # wav = wav[:rate * int(self.duration)]
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
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
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
    x = layers.Lambda(spectrogram2wav)(x)
    return keras.Model(inputs, x, name="synth")


def spectrogram2wav(spectrogram, n_iter=60, n_fft=1024,
                    win_length=1024,
                    hop_length=1024 // 4):
    '''Converts spectrogram into a waveform using Griffin-lim's raw.
    '''
    spectrogram = tf.transpose(spectrogram, perm=(0, 3, 1, 2))

    spectrogram = tf.cast(spectrogram, dtype=tf.complex64)  # [t, f]
    X_best = tf.identity(spectrogram)
    for i in range(n_iter):
        X_t = tf.signal.inverse_stft(X_best, win_length, hop_length, n_fft)
        est = tf.signal.stft(X_t, win_length, hop_length, n_fft, pad_end=False)  # (1, T, n_fft/2+1)
        phase = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)  # [t, f]
        X_best = spectrogram * phase  # [t, t]
    X_t = tf.signal.inverse_stft(X_best, win_length, hop_length, n_fft)
    y = tf.math.real(X_t)
    y = tf.transpose(y, perm=(0, 2, 1))
    return y


def get_model(latent_dim=8, sr=44100, duration=3.0):
    input_shape = (int(sr * duration), 1)
    encoder_inputs = keras.Input(shape=input_shape)
    x = kapre.STFT(input_shape=input_shape, n_fft=1024)(encoder_inputs)
    x = kapre.Magnitude()(x)
    x = layers.Lambda(lambda m: (m - tf.reduce_min(m)) / (tf.reduce_max(m) - tf.reduce_min(m)))(x)
    stft_out = layers.Lambda(lambda m: tf.image.resize_with_crop_or_pad(m, 513, 513))(x)
    stft_model = keras.Model(encoder_inputs, stft_out, name='stft')

    img_inputs = keras.Input(shape=(513, 513, 1))
    x = layers.TimeDistributed(layers.Conv1D(64, 3, padding="same"))(img_inputs)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.TimeDistributed(layers.Conv1D(64, 3, padding="same"))(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.TimeDistributed(layers.Conv1D(64, 3, padding="same"))(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.TimeDistributed(layers.Conv1D(64, 3, padding="same"))(x)
    x = layers.TimeDistributed(layers.Conv1D(1, 3, padding="same"))(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean", activation="tanh")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var", activation="tanh")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(img_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(64 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((64, 64, 1))(x)
    x = layers.TimeDistributed(layers.Conv1DTranspose(64, 3, padding="same"))(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D()(x)
    x = layers.TimeDistributed(layers.Conv1DTranspose(64, 3, padding="same"))(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D()(x)
    x = layers.TimeDistributed(layers.Conv1DTranspose(64, 3, padding="same"))(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D()(x)
    x = layers.ZeroPadding2D(padding=[(0, 1), (0, 1)])(x)
    x = layers.TimeDistributed(layers.Conv1DTranspose(64, 3, padding="same"))(x)
    x = layers.ReLU()(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = VAE(stft_model, encoder, decoder)

    return vae


def pad_up_to(t, max_in_dims, constant_values):
    s = tf.shape(t)
    paddings = [[0, m - s[i]] for (i, m) in enumerate(max_in_dims)]
    return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)


if __name__ == '__main__':
    stft_model_path = os.path.join(os.path.dirname(__file__), 'models', 'stft_mod_v{}'.format(1))
    enc_model_path = os.path.join(os.path.dirname(__file__), 'models', 'enc_mod_v{}'.format(1))
    dec_model_path = os.path.join(os.path.dirname(__file__), 'models', 'dec_mod_v{}'.format(1))

    path = os.path.join(os.path.dirname(__file__), 'samples')
    sr = 44100
    duration = 3.0
    batch_size = 4

    sequence = SoundSequence(path, sr=sr, duration=duration, batch_size=batch_size)

    autoencoder = None
    if os.path.exists(stft_model_path) and os.path.exists(enc_model_path) and os.path.exists(dec_model_path):
        autoencoder = VAE(
            tf.keras.models.load_model(stft_model_path, compile=False),
            tf.keras.models.load_model(enc_model_path, compile=False),
            tf.keras.models.load_model(dec_model_path, compile=False)
        )
    else:
        autoencoder = get_model(latent_dim=8, sr=sr, duration=duration)

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
    # # stft = tf.image.per_image_standardization(stft)
    # stft = (stft - tf.reduce_min(stft)) / (tf.reduce_max(stft) - tf.reduce_min(stft))
    # # repro = stft_to_wav_model()(stft)
    # repro = spectrogram2wav(stft)
    # # repro = tf.transpose(repro)
    # repro = repro[0]
    # repro = librosa.util.normalize(repro)
    # repro = tf.audio.encode_wav(repro, 44100)
    # tf.io.write_file('reproduction.wav', repro)

    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005))
    autoencoder.fit(sequence, epochs=100)

    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'models')):
        os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)

    autoencoder.stft.save(stft_model_path, save_format='tf', include_optimizer=False)
    autoencoder.encoder.save(enc_model_path, save_format='tf', include_optimizer=False)
    autoencoder.decoder.save(dec_model_path, save_format='tf', include_optimizer=False)

    synth = get_synth_model(autoencoder.decoder)
    synth.summary()
    #
    random = tf.random.normal([5, 8])
    wavs = synth.predict_on_batch(random)

    i = 0
    for wav in wavs:
        wav = librosa.util.normalize(wav)
        wav = tf.audio.encode_wav(wav, sr)
        tf.io.write_file('output-{}.wav'.format(i), wav)
        i = i + 1
