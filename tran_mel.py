import os
from datetime import datetime
import tensorflow as tf
import librosa

from generators import SoundSequence
from callbacks import WaveformCallback
from models import get_mfcc_autoencoder

if __name__ == '__main__':
    logdir = os.path.join(os.path.dirname(__file__), 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))

    mel_autonencoder_path = os.path.join(os.path.dirname(__file__), 'models', 'mel_autoenc_mod_v{}'.format(1))

    path = os.path.join(os.path.dirname(__file__), 'samples')
    sr = 44100
    duration = 1.0
    batch_size = 16
    epochs = 200

    sequence = SoundSequence(path, sr=sr, duration=duration, batch_size=batch_size, as_autoencoder=True)

    autoencoder = None
    if os.path.exists(mel_autonencoder_path):
        autoencoder = tf.keras.models.load_model(mel_autonencoder_path, compile=False),
    else:
        autoencoder = get_mfcc_autoencoder(sr, duration)

    autoencoder.summary()

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mae')
    autoencoder.fit(sequence, epochs=epochs, callbacks=[
        WaveformCallback(sequence, sr=sr, logdir=logdir),
        tf.keras.callbacks.TensorBoard(log_dir=logdir)
    ])

    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'models')):
        os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)

    autoencoder.save(mel_autonencoder_path, save_format='tf', include_optimizer=False)

    # synth = get_synth_model(autoencoder.decoder, input_shape=(latent_dim,))
    # synth.summary()
    # #
    # random = tf.random.normal([5, latent_dim])
    # wavs = synth.predict_on_batch(random)
    #
    # i = 0
    # for wav in wavs:
    #     wav = librosa.util.normalize(wav)
    #     wav = tf.cast(wav, tf.float32)
    #     wav = tf.audio.encode_wav(wav, sr)
    #     tf.io.write_file('output-{}.wav'.format(i), wav)
    #     i = i + 1