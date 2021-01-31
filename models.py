import kapre
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import mag_phase_to_complex


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the STFT."""

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


def residual_module(layer_in, n_filters):
    merge_input = layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = tf.keras.layers.Conv2D(n_filters, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(
            layer_in)
    # conv1
    conv1 = tf.keras.layers.Conv2D(n_filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    # conv2
    conv2 = tf.keras.layers.Conv2D(n_filters, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
    # conv3
    conv3 = tf.keras.layers.Conv2D(n_filters, (3, 3), padding='same', activation='linear',
                                   kernel_initializer='he_normal')(conv2)
    # add filters, assumes filters/channels last
    layer_out = tf.keras.layers.Add()([conv3, conv2, merge_input])
    # activation function
    layer_out = tf.keras.layers.BatchNormalization()(layer_out)
    layer_out = tf.keras.layers.Activation('relu')(layer_out)
    return layer_out


def residual_transpose_module(layer_in, n_filters):
    merge_input = layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = tf.keras.layers.Conv2DTranspose(n_filters, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(
            layer_in)
    # conv1
    conv1 = tf.keras.layers.Conv2DTranspose(n_filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    # conv2
    conv2 = tf.keras.layers.Conv2DTranspose(n_filters, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
    # conv3
    conv3 = tf.keras.layers.Conv2DTranspose(n_filters, (3, 3), padding='same', activation='linear',
                                            kernel_initializer='he_normal')(conv2)
    # add filters, assumes filters/channels last
    layer_out = tf.keras.layers.Add()([conv3, conv2, merge_input])
    # activation function
    layer_out = tf.keras.layers.BatchNormalization()(layer_out)
    layer_out = tf.keras.layers.Activation('relu')(layer_out)
    return layer_out


def get_synth_model(decoder, input_shape=(8,), n_fft=2048):
    inputs = keras.Input(shape=input_shape)
    x = decoder(inputs)
    x = layers.Lambda(mag_phase_to_complex)(x)
    x = kapre.InverseSTFT(n_fft=n_fft)(x)
    x = layers.Lambda(lambda h: tf.cast(h, tf.float32))(x)
    return keras.Model(inputs, x, name="synth")


def get_model(latent_dim=8, sr=44100, duration=1.0, spectrogram_shape=(80, 1025), n_fft=2048):
    input_shape = (int(sr * duration), 1)
    encoder_inputs = keras.Input(shape=input_shape)
    x = kapre.composed.get_stft_mag_phase(input_shape, n_fft=n_fft, return_decibel=False)(encoder_inputs)
    x = layers.experimental.preprocessing.Normalization(name='normalizer')(x)
    stft_out = layers.Lambda(lambda m: tf.image.resize_with_crop_or_pad(m, spectrogram_shape[0], spectrogram_shape[1]))(
        x)
    stft_model = keras.Model(encoder_inputs, stft_out, name='stft')

    img_inputs = keras.Input(shape=(spectrogram_shape[0], spectrogram_shape[1], 2))
    x = residual_module(img_inputs, 8)
    x = layers.AveragePooling2D()(x)
    x = residual_module(x, 16)
    x = layers.AveragePooling2D()(x)
    x = residual_module(x, 32)
    x = layers.AveragePooling2D()(x)
    x = residual_module(x, 64)
    x = layers.AveragePooling2D()(x)
    x = residual_module(x, 128)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean", activation=None)(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var", activation=None)(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(img_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(5 * 64 * 128, activation='relu')(latent_inputs)
    x = layers.Reshape((5, 64, 128))(x)
    x = residual_transpose_module(x, 128)
    x = layers.UpSampling2D(interpolation="nearest")(x)
    x = residual_transpose_module(x, 64)
    x = layers.UpSampling2D(interpolation="nearest")(x)
    x = residual_transpose_module(x, 32)
    x = layers.UpSampling2D(interpolation="nearest")(x)
    x = residual_transpose_module(x, 16)
    x = layers.UpSampling2D(interpolation="nearest")(x)
    x = layers.ZeroPadding2D(padding=[(0, 0), (0, 1)])(x)
    x = residual_transpose_module(x, 8)
    decoder_outputs = layers.Conv2DTranspose(2, 3, activation=None, padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = VAE(stft_model, encoder, decoder)

    return vae


if __name__ == '__main__':
    vae = get_model(latent_dim=16)
    vae.encoder.summary()
    vae.decoder.summary()
