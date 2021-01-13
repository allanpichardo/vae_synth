from models import get_synth_model, get_model
import librosa
import tensorflow as tf
import tensorflowjs as tfjs
import os

decoder = tf.keras.models.load_model('models/dec_mod_v1')
synth = get_synth_model(decoder)

random = tf.random.normal([25, 8])
wavs = synth.predict_on_batch(random)

i = 0
for wav in wavs:
    wav = librosa.util.normalize(wav)
    wav = tf.cast(wav, tf.float32)
    wav = tf.audio.encode_wav(wav, 44010)
    tf.io.write_file('output-{}.wav'.format(i), wav)
    i = i + 1

tf.keras.models.save_model(synth, os.path.join(os.path.dirname(__file__), 'synth_model'), overwrite=True, save_format='tf')
tfjs.converters.convert_tf_saved_model(
    os.path.join(os.path.dirname(__file__), 'synth_model'),
    os.path.join(os.path.dirname(__file__), 'js_synth_model'),
    control_flow_v2=True, experiments=True
)