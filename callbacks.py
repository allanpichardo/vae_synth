import kapre
import tensorflow as tf
from datetime import datetime
import librosa
import matplotlib.pyplot as plt
import io

from utils import mag_phase_to_complex


class SpectrogramCallback(tf.keras.callbacks.Callback):

    def __init__(self, soundsequence, sr=44100, logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"), n_fft=2048):
        super().__init__()
        self.soundequence = soundsequence
        self.logdir = logdir
        self.sr = sr
        self.n_fft = 2048
        self.mean = []
        self.variance = []

    def on_train_begin(self, logs=None):
        print("Initializing normalize layer...")

        should_reset = True
        for x, y in self.soundequence:
            spec_x = self.model.stft(x)
            if should_reset:
                self.model.stft.get_layer('normalizer').reset_state()
            self.model.stft.get_layer('normalizer').adapt(spec_x)
            should_reset = False

        self.mean = tf.squeeze(self.model.stft.get_layer('normalizer').mean)
        self.variance = tf.squeeze(self.model.stft.get_layer('normalizer').variance)

        print('Mean: {} | Var: {}'.format(self.mean,
                                          self.variance))

    def normalize(self, x):
        return (x - tf.reduce_min(x, axis=[0, 1, 2])) / (tf.reduce_max(x, axis=[0, 1, 2]) - tf.reduce_min(x, axis=[0, 1, 2]))

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.soundequence.__getitem__(0)

        spec_x = self.model.stft(x)
        u, v, embedding = self.model.encoder(spec_x)
        spec_y = self.model.decoder(embedding)
        audio_y = kapre.InverseSTFT(n_fft=self.n_fft)(mag_phase_to_complex(spec_y, self.mean, self.variance))

        mag_x = kapre.MagnitudeToDecibel()(kapre.Magnitude()(mag_phase_to_complex(spec_x, self.mean, self.variance)))
        mag_y = kapre.MagnitudeToDecibel()(kapre.Magnitude()(mag_phase_to_complex(spec_y, self.mean, self.variance)))

        file_writer = tf.summary.create_file_writer(self.logdir)

        with file_writer.as_default():
            tf.summary.audio("Sample Input", tf.cast(x, tf.float32), self.sr, step=epoch, max_outputs=5,
                             description="Audio sample input")
            tf.summary.image("STFT Input", self.normalize(mag_x), step=epoch, max_outputs=5,
                             description="Spectrogram input")
            tf.summary.image("STFT Reconstruction", self.normalize(mag_y), step=epoch, max_outputs=5,
                             description="Spectrogram output")
            tf.summary.audio("Sample Reconstruction", tf.cast(librosa.util.normalize(audio_y), tf.float32), self.sr,
                             step=epoch, max_outputs=5,
                             description="Synthesized audio")


class WaveformCallback(tf.keras.callbacks.Callback):

    def __init__(self, soundsequence, sr=44100, logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")):
        super().__init__()
        self.soundequence = soundsequence
        self.logdir = logdir
        self.sr = sr

    def gen_plot(self, wave):
        """Create a pyplot plot and save to buffer."""
        w = tf.squeeze(wave, axis=-1)
        plt.figure()
        plt.plot(w[0].numpy())
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = tf.image.decode_png(buf.getvalue(), channels=4)
        img = tf.expand_dims(img, axis=0)
        plt.close()
        return img

    def normalize(self, x):
        return (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.soundequence.__getitem__(0)

        audio_y = self.model(x)
        file_writer = tf.summary.create_file_writer(self.logdir)

        with file_writer.as_default():
            tf.summary.audio("Sample Input", tf.cast(x, tf.float32), self.sr, step=epoch, max_outputs=2,
                             description="Audio sample input")
            tf.summary.audio("Sample Reconstruction", tf.cast(librosa.util.normalize(audio_y), tf.float32), self.sr,
                             step=epoch, max_outputs=2,
                             description="Synthesized audio")
            tf.summary.image("Waveform Input", self.gen_plot(x), step=epoch, max_outputs=1)
            tf.summary.image("Waveform Reconstruction", self.gen_plot(audio_y), step=epoch, max_outputs=1)