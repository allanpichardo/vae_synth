# encoding: utf-8
import tensorflow as tf

"""
TensorFlow implementation of Griffin-lim Algorithm for voice reconstruction
https://github.com/candlewill/Griffin_lim
"""

num_mels = 80
num_freq = 513
sample_rate = 22050
frame_length_ms = 50
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
max_abs_value = 4
power = 1.5
fft_size = 1024
hop_size = 256

# Eval:
griffin_lim_iters = 120


def GriffinLim():
    return tf.keras.layers.Lambda(inv_spectrogram)


def STFTNormalize():
    return tf.keras.layers.Lambda(_normalize)


# TF
def spectrogram2wav(spectrogram, n_iter=griffin_lim_iters, n_fft=1024,
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


def inv_spectrogram(spectrogram):
    S = _denormalize(spectrogram)
    S = _db_to_amp(S + ref_level_db)  # Convert back to linear
    return spectrogram2wav(S ** power)  # Reconstruct phase


def _denormalize(D):
    clipped = tf.clip_by_value(D, -max_abs_value, max_abs_value)
    return (((clipped + max_abs_value) * -min_level_db / (
            2 * max_abs_value)) + min_level_db)


def _normalize(S):
    return tf.clip_by_value((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value,
                            -max_abs_value,
                            max_abs_value
                            )


def _db_to_amp(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _inv_preemphasis(x):
    N = tf.shape(x)[0]
    i = tf.constant(0)
    W = tf.zeros(shape=tf.shape(x), dtype=tf.float32)

    def condition(i, y):
        return tf.less(i, N)

    def body(i, y):
        tmp = tf.slice(x, [0], [i + 1])
        tmp = tf.concat([tf.zeros([N - i - 1]), tmp], -1)
        y = preemphasis * y + tmp
        i = tf.add(i, 1)
        return [i, y]

    final = tf.while_loop(condition, body, [i, W])

    y = final[1]

    return y
