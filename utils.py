import tensorflow as tf


def mag_phase_to_complex(x, mean=[1.8400466, 0.00264591], var=[163.19633, 3.3680449]):
    x = (x * tf.sqrt(var)) + mean
    m, p, d = tf.split(x, 3, axis=3)
    real = m * tf.cos(p)
    imag = m * tf.sin(p)
    return tf.complex(real, imag)


def pad_up_to(t, max_in_dims, constant_values):
    s = tf.shape(t)
    paddings = [[0, m - s[i]] for (i, m) in enumerate(max_in_dims)]
    return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)