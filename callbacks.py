import tensorflow as tf
import datetime
from models import spectrogram2wav
from tensorboard.plugins import projector
import os


class SpectrogramCallback(tf.keras.callbacks.Callback):

    def __init__(self, soundsequence, sr=44100, logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")):
        super().__init__()
        self.soundequence = soundsequence
        self.logdir = logdir
        self.sr = sr

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.soundequence.__get_item__(0)

        spec_x = self.model.stft(x)
        embedding = self.encoder(spec_x)
        spec_y = self.decoder(embedding)
        audio_y = spectrogram2wav(spec_y)

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
