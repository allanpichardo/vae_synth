import tensorflow as tf


def mag_phase_to_complex(x, mean=[0.1928852, 0.00132339], var=[4.575152, 1.8032672]):
    x = (x * tf.sqrt(var)) + mean
    m, p = tf.split(x, 2, axis=3)
    real = m * tf.cos(p)
    imag = m * tf.sin(p)
    return tf.complex(real, imag)


def pad_up_to(t, max_in_dims, constant_values):
    s = tf.shape(t)
    paddings = [[0, m - s[i]] for (i, m) in enumerate(max_in_dims)]
    return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)