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
from griffin_lim import GriffinLim, STFTNormalize, STFTDenormalize, DBToAmp

N_FFT = 512


class SpectrogramCallback(tf.keras.callbacks.Callback):

    def __init__(self, soundsequence, sr=44100, logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")):
        super().__init__()
        self.soundequence = soundsequence
        self.logdir = logdir
        self.sr = sr

    def on_train_begin(self, logs=None):
        print("Initializing normalize layer...")

        should_reset = True
        for x, y in self.soundequence:
            spec_x = self.model.stft(x)
            self.model.stft.get_layer('normalizer').adapt(spec_x, reset_state=should_reset)
            should_reset = False

        print('Mean: {} | Var: {}'.format(self.model.stft.get_layer('normalizer').mean,
                                          self.model.stft.get_layer('normalizer').variance))

    def normalize(self, x):
        return (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.soundequence.__getitem__(0)

        spec_x = self.model.stft(x)
        embedding = self.model.encoder(spec_x)
        spec_y = self.model.decoder(embedding)
        audio_y = kapre.InverseSTFT(n_fft=N_FFT)(mag_phase_to_complex(spec_y))

        mag_x = kapre.MagnitudeToDecibel()(kapre.Magnitude()(mag_phase_to_complex(spec_x)))
        mag_y = kapre.MagnitudeToDecibel()(kapre.Magnitude()(mag_phase_to_complex(spec_y)))

        file_writer = tf.summary.create_file_writer(self.logdir)

        with file_writer.as_default():
            tf.summary.audio("Sample Input", x, self.sr, step=epoch, max_outputs=5, description="Audio sample input")
            tf.summary.image("STFT Input", self.normalize(mag_x), step=epoch, max_outputs=5, description="Spectrogram input")
            tf.summary.image("STFT Reconstruction", self.normalize(mag_y), step=epoch, max_outputs=5,
                             description="Spectrogram output")
            tf.summary.audio("Sample Reconstruction", librosa.util.normalize(audio_y), self.sr, step=epoch, max_outputs=5,
                             description="Synthesized audio")


class SoundSequence(tf.keras.utils.Sequence):

    def __init__(self, music_path, n_fft=N_FFT, sr=44100, duration=2.0, batch_size=32, shuffle=True):
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

            # reconstruction_loss = tf.sqrt(
            #     tf.divide(
            #         tf.reduce_sum(tf.square(stft_out - reconstruction)),
            #         tf.reduce_sum(tf.square(stft_out))
            #     )
            # )
            reconstruction_loss = tf.keras.losses.MeanAbsoluteError()(stft_out, reconstruction)

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


def mag_phase_to_complex(x):
    m, p = tf.split(x, 2, axis=3)
    real = m * tf.cos(p)
    imag = m * tf.sin(p)
    return tf.complex(real, imag)


def get_synth_model(decoder, input_shape=(8,)):
    inputs = keras.Input(shape=input_shape)
    x = decoder(inputs)
    x = layers.Lambda(mag_phase_to_complex)(x)
    x = kapre.InverseSTFT(n_fft=N_FFT)(x)
    return keras.Model(inputs, x, name="synth")


def get_model(latent_dim=8, sr=44100, duration=3.0):
    input_shape = (int(sr * duration), 1)
    encoder_inputs = keras.Input(shape=input_shape)
    x = kapre.composed.get_stft_mag_phase(input_shape, n_fft=N_FFT, return_decibel=False)(encoder_inputs)
    x = layers.experimental.preprocessing.Normalization(name='normalizer')(x)
    stft_out = layers.Lambda(lambda m: tf.image.resize_with_crop_or_pad(m, 257, 257))(x)
    stft_model = keras.Model(encoder_inputs, stft_out, name='stft')

    img_inputs = keras.Input(shape=(257, 257, 2))
    x = layers.Conv2D(32, 3, padding="same")(img_inputs)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.Conv2D(1, 3, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean", activation=None)(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var", activation=None)(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(img_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(32 * 32, activation="relu")(latent_inputs)
    x = layers.Reshape((32, 32, 1))(x)
    x = layers.Conv2DTranspose(32, 3, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.UpSampling2D(interpolation="bilinear")(x)
    x = layers.Conv2DTranspose(32, 3, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.UpSampling2D(interpolation="bilinear")(x)
    x = layers.Conv2DTranspose(32, 3, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.UpSampling2D(interpolation="bilinear")(x)
    x = layers.ZeroPadding2D(padding=[(0, 1), (0, 1)])(x)
    x = layers.Conv2DTranspose(32, 3, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, padding="same")(x)
    x = layers.LeakyReLU()(x)
    decoder_outputs = layers.Conv2DTranspose(2, 3, activation=None, padding="same")(x)
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
    latent_dim = 8

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
    # repro = kapre.InverseSTFT()(mag_phase_to_complex(stft))
    # # repro = GriffinLim()(stft)
    # repro = repro[0]
    # repro = librosa.util.normalize(repro)
    # repro = tf.audio.encode_wav(repro, rate)
    # tf.io.write_file('reproduction.wav', repro)

    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00005))
    autoencoder.fit(sequence, epochs=1000, callbacks=[
        SpectrogramCallback(sequence, sr=sr),
        tf.keras.callbacks.TensorBoard(log_dir=logdir, embeddings_freq=1)
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
