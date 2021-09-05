import os
from glob import glob
import numpy as np
import librosa
import tensorflow as tf


class SoundSequence(tf.keras.utils.Sequence):

    def __init__(self, music_path, sr=44100, duration=2.0, batch_size=32, shuffle=True, as_autoencoder=False):
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
        self.as_autoencoder = as_autoencoder

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
            wav = librosa.util.normalize(wav)
            wav = tf.convert_to_tensor(wav)
            # wav = tf.cast(wav, tf.float16)
            wav = tf.expand_dims(wav, 1)
            wav = self.pad_up_to(wav, [rate * int(self.duration), 1], 0)
            X.append(wav)
            Y.append(tf.convert_to_tensor(label))

        X = tf.stack(X)
        Y = tf.stack(Y)

        return (X, Y) if not self.as_autoencoder else (X, X)

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


if __name__ == '__main__':
    seq = SoundSequence('/Users/allanpichardo/PycharmProjects/audio-generation-autoencoder/samples', duration=1.0)

    for x, y in seq:
        for wav in x:
            print(tf.reduce_min(wav))
            print(tf.reduce_max(wav))