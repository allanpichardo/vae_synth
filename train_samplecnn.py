import os
from datetime import datetime
import tensorflow as tf
import librosa

from generators import SoundSequence
from callbacks import WaveformCallback
from models import get_sample_model, get_sample_synth_model, SampleVAE

if __name__ == '__main__':
    logdir = os.path.join(os.path.dirname(__file__), 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))

    enc_model_path = os.path.join(os.path.dirname(__file__), 'models', 'samp_enc_mod_v{}'.format(1))
    dec_model_path = os.path.join(os.path.dirname(__file__), 'models', 'samp_dec_mod_v{}'.format(1))

    path = os.path.join(os.path.dirname(__file__), 'samples')
    sr = 44100
    duration = 1.0
    batch_size = 8
    latent_dim = 8
    epochs = 200

    tf.keras.backend.set_floatx('float16')

    sequence = SoundSequence(path, sr=sr, duration=duration, batch_size=batch_size)

    autoencoder = None
    if os.path.exists(enc_model_path) and os.path.exists(dec_model_path):
        autoencoder = SampleVAE(
            tf.keras.models.load_model(enc_model_path, compile=False),
            tf.keras.models.load_model(dec_model_path, compile=False)
        )
    else:
        autoencoder = get_sample_model(latent_dim=latent_dim, sr=sr, duration=duration)

    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    autoencoder.fit(sequence, epochs=epochs, callbacks=[
        WaveformCallback(sequence, sr=sr, logdir=logdir),
        tf.keras.callbacks.TensorBoard(log_dir=logdir, embeddings_freq=1)
    ])

    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'models')):
        os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)

    autoencoder.encoder.save(enc_model_path, save_format='tf', include_optimizer=False)
    autoencoder.decoder.save(dec_model_path, save_format='tf', include_optimizer=False)

    synth = get_sample_synth_model(autoencoder.decoder, input_shape=(latent_dim,))
    synth.summary()
    #
    random = tf.random.normal([5, latent_dim])
    wavs = synth.predict_on_batch(random)

    i = 0
    for wav in wavs:
        wav = librosa.util.normalize(wav)
        wav = tf.cast(wav, tf.float32)
        wav = tf.audio.encode_wav(wav, sr)
        tf.io.write_file('output-{}.wav'.format(i), wav)
        i = i + 1