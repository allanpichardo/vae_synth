import kapre
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

from utils import mag_phase_to_complex


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the STFT."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class SampleVAE(keras.Model):

    def call(self, inputs, training=None, mask=None):
        u, v, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(
            v - tf.square(u) - tf.exp(v) + 1
        )
        self.add_loss(kl_loss)

        return reconstructed

    def __init__(self, encoder, decoder, input_shape=(44100, 1), sr=44100, **kwargs):
        super(SampleVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.stft = kapre.composed.get_melspectrogram_layer(input_shape=input_shape, sample_rate=sr)
        # self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        # self.reconstruction_loss_tracker = keras.metrics.Mean(
        #     name="reconstruction_loss"
        # )
        # self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    # @property
    # def metrics(self):
    #     return [
    #         self.total_loss_tracker,
    #         self.reconstruction_loss_tracker,
    #         self.kl_loss_tracker,
    #     ]
    #
    # def root_mean_squared_error(self, y_true, y_pred):
    #     return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    # def train_step(self, data):
    #     if isinstance(data, tuple):
    #         data = data[0]
    #     with tf.GradientTape() as tape:
    #         z_mean, z_log_var, z = self.encoder(data)
    #         reconstruction = self.decoder(z)
    #
    #         spec_x = self.stft(data)
    #         spec_y = self.stft(reconstruction)
    #
    #         # reconstruction_loss = self.root_mean_squared_error(spec_x, spec_y)
    #         # reconstruction_loss = tf.keras.losses.MeanSquaredError()(spec_x, spec_y)
    #         reconstruction_loss = 1
    #
    #         coefficient = 0.0001
    #         kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    #         kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) * coefficient
    #         total_loss = reconstruction_loss + kl_loss
    #
    #     grads = tape.gradient(total_loss, self.trainable_weights)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    #     self.total_loss_tracker.update_state(total_loss)
    #     self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    #     self.kl_loss_tracker.update_state(kl_loss)
    #     return {
    #         "loss": self.total_loss_tracker.result(),
    #         "reconstruction_loss": self.reconstruction_loss_tracker.result(),
    #         "kl_loss": self.kl_loss_tracker.result(),
    #     }


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

    def root_mean_squared_error(self, y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            stft_out = self.stft(data)
            z_mean, z_log_var, z = self.encoder(stft_out)
            reconstruction = self.decoder(z)

            reconstruction_loss = self.root_mean_squared_error(stft_out, reconstruction)

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


class SampleConv(tf.keras.layers.Layer):

    def __init__(self, filters, **kwargs):
        super(SampleConv, self).__init__(**kwargs)
        self.filters = filters
        self.conv = tf.keras.layers.Conv1D(self.filters, 3, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.pooling = tf.keras.layers.AveragePooling1D()

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x


class SampleConvTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(SampleConvTranspose, self).__init__(**kwargs)
        self.filters = filters
        self.conv = tf.keras.layers.Conv1DTranspose(self.filters, 3, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.upsampling = tf.keras.layers.UpSampling1D()

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upsampling(x)
        return x


def residual_module(layer_in, n_filters):
    merge_input = layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = tf.keras.layers.Conv2D(n_filters, (1, 1), padding='same', activation='relu',
                                             kernel_initializer='he_normal')(
            layer_in)
    # conv1
    conv1 = tf.keras.layers.Conv2D(n_filters, (3, 3), padding='same', activation='relu',
                                   kernel_initializer='he_normal')(layer_in)
    # conv2
    conv2 = tf.keras.layers.Conv2D(n_filters, (3, 3), padding='same', activation='linear',
                                   kernel_initializer='he_normal')(conv1)
    # conv3
    # conv3 = tf.keras.layers.Conv2D(n_filters, (3, 3), padding='same', activation='linear',
    #                                kernel_initializer='he_normal')(conv2)
    # add filters, assumes filters/channels last
    layer_out = tf.keras.layers.Add()([conv2, merge_input])
    # activation function
    # layer_out = tf.keras.layers.Conv2D(n_filters, (1, 1), padding='same', kernel_initializer='he_normal')(
    #     layer_out)
    # layer_out = tf.keras.layers.BatchNormalization()(layer_out)
    layer_out = tf.keras.layers.Activation('relu')(layer_out)
    return layer_out


def residual_transpose_module(layer_in, n_filters):
    merge_input = layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = tf.keras.layers.Conv2DTranspose(n_filters, (1, 1), padding='same', activation='relu',
                                                      kernel_initializer='he_normal')(
            layer_in)
    # conv1
    conv1 = tf.keras.layers.Conv2DTranspose(n_filters, (3, 3), padding='same', activation='relu',
                                            kernel_initializer='he_normal')(layer_in)
    # conv2
    conv2 = tf.keras.layers.Conv2DTranspose(n_filters, (3, 3), padding='same', activation='linear',
                                            kernel_initializer='he_normal')(conv1)
    # conv3
    # conv3 = tf.keras.layers.Conv2DTranspose(n_filters, (3, 3), padding='same', activation='linear',
    #                                         kernel_initializer='he_normal')(conv2)
    # add filters, assumes filters/channels last
    layer_out = tf.keras.layers.Add()([conv2, merge_input])
    # activation function
    # layer_out = tf.keras.layers.Conv2DTranspose(n_filters, (1, 1), padding='same', kernel_initializer='he_normal')(layer_out)
    # layer_out = tf.keras.layers.BatchNormalization()(layer_out)
    layer_out = tf.keras.layers.Activation('relu')(layer_out)
    return layer_out


def conv_1d_block(filters, inputs):
    x = tf.keras.layers.Conv1D(filters, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.AveragePooling1D()(x)
    return tf.keras.Model(inputs, x, name='sampleConv')


def conv_1d_transpose_block(filters, inputs):
    x = tf.keras.layers.Conv1DTranspose(filters, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling1D()(x)
    return tf.keras.Model(inputs, x, name='sampleDeconv')


def get_synth_model(decoder, input_shape=(8,), n_fft=2048):
    inputs = keras.Input(shape=input_shape)
    x = decoder(inputs)
    x = layers.Lambda(mag_phase_to_complex)(x)
    x = kapre.InverseSTFT(n_fft=n_fft)(x)
    x = layers.Lambda(lambda h: tf.cast(h, tf.float32))(x)
    return keras.Model(inputs, x, name="synth")


def get_sample_synth_model(decoder, input_shape=(8,)):
    inputs = keras.Input(shape=input_shape)
    x = decoder(inputs)
    x = layers.Lambda(lambda h: tf.cast(h, tf.float32))(x)
    return keras.Model(inputs, x, name="synth")


def get_sample_model(latent_dim=8, sr=44100, duration=1.0):
    input_shape = (int(sr * duration), 1)

    encoder_inputs = tf.keras.layers.Input(shape=input_shape)
    x = SampleConv(8)(encoder_inputs)
    x = SampleConv(8)(x)
    x = SampleConv(16)(x)
    x = SampleConv(16)(x)
    x = SampleConv(32)(x)
    x = SampleConv(32)(x)
    x = tf.keras.layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean", activation=None)(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var", activation=None)(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(689 * 32)(latent_inputs)
    x = tf.keras.layers.Reshape((689, 32))(x)
    x = SampleConvTranspose(32)(x)
    x = SampleConvTranspose(32)(x)
    x = SampleConvTranspose(16)(x)
    x = SampleConvTranspose(16)(x)
    x = tf.keras.layers.ZeroPadding1D((0, 1))(x)
    x = SampleConvTranspose(8)(x)
    x = SampleConvTranspose(8)(x)
    decoder_outputs = tf.keras.layers.Conv1DTranspose(1, 3, padding='same', activation=None)(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return SampleVAE(encoder, decoder)


def get_model(latent_dim=8, sr=44100, duration=1.0, spectrogram_shape=(80, 1025), n_fft=2048):
    input_shape = (int(sr * duration), 1)
    encoder_inputs = keras.Input(shape=input_shape)
    x = kapre.composed.get_stft_mag_phase(input_shape, n_fft=n_fft, return_decibel=False)(encoder_inputs)
    x = layers.Lambda(lambda m: tf.image.resize_with_crop_or_pad(m, spectrogram_shape[0], spectrogram_shape[1]))(x)

    x2 = kapre.composed.get_melspectrogram_layer(input_shape, n_fft=n_fft, sample_rate=sr, n_mels=spectrogram_shape[1], return_decibel=True)(encoder_inputs)
    x2 = layers.Lambda(lambda m: tf.image.resize_with_crop_or_pad(m, spectrogram_shape[0], spectrogram_shape[1]))(x2)

    x = tf.keras.layers.Concatenate()([x, x2])
    x = layers.experimental.preprocessing.Normalization(name='normalizer')(x)
    stft_out = layers.Lambda(lambda m: tf.image.resize_with_crop_or_pad(m, spectrogram_shape[0], spectrogram_shape[1]))(
        x)
    stft_model = keras.Model(encoder_inputs, stft_out, name='stft')

    img_inputs = keras.Input(shape=(spectrogram_shape[0], spectrogram_shape[1], 3))
    x = residual_module(img_inputs, 16)
    x = residual_module(x, 16)
    x = layers.AveragePooling2D()(x)
    x = residual_module(x, 16)
    x = residual_module(x, 16)
    x = layers.AveragePooling2D()(x)
    x = residual_module(x, 16)
    x = residual_module(x, 16)
    x = layers.AveragePooling2D()(x)
    x = residual_module(x, 16)
    x = residual_module(x, 16)
    x = layers.AveragePooling2D()(x)
    x = residual_module(x, 16)
    x = residual_module(x, 16)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean", activation=None)(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var", activation=None)(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(img_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(5 * 64 * 16, activation='relu')(latent_inputs)
    x = layers.Reshape((5, 64, 16))(x)
    x = residual_transpose_module(x, 16)
    x = residual_transpose_module(x, 16)
    x = layers.UpSampling2D(interpolation="nearest")(x)
    x = residual_transpose_module(x, 16)
    x = residual_transpose_module(x, 16)
    x = layers.UpSampling2D(interpolation="nearest")(x)
    x = residual_transpose_module(x, 16)
    x = residual_transpose_module(x, 16)
    x = layers.UpSampling2D(interpolation="nearest")(x)
    x = residual_transpose_module(x, 16)
    x = residual_transpose_module(x, 16)
    x = layers.UpSampling2D(interpolation="nearest")(x)
    x = layers.ZeroPadding2D(padding=[(0, 0), (0, 1)])(x)
    x = residual_transpose_module(x, 16)
    x = residual_transpose_module(x, 16)
    decoder_outputs = layers.Conv2DTranspose(3, 3, activation=None, padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = VAE(stft_model, encoder, decoder)

    return vae


if __name__ == '__main__':
    m = get_model()
    m.build((32, 44100, 1))
    m.stft.summary()
    # test = m.decoder(tf.expand_dims(tf.convert_to_tensor([0, 0, 0, 0, 0, 0, 0, 0]), 0))
    # blah = m.encoder(test)
    # print(test)
    # foo = m.predict(test)
    # print(foo)
