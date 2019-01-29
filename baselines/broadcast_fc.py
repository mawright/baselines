import tensorflow as tf
from baselines.a2c.utils import fc


def broadcast_fc(x, scope, units, use_bias=True, init_scale=1.0, init_bias=0.0):
    if len(x.shape) == 2:
        return fc(x, scope, units, init_scale=init_scale, init_bias=init_bias)
    with tf.variable_scope(scope):
        in_shape_tensor = tf.shape(x)
        in_units = x.shape[-1].value

        w = tf.get_variable('w', [in_units, units],
                            initializer=tf.initializers.orthogonal(init_scale))

        reshaped_x = tf.reshape(x, [-1, in_units])

        out = tf.matmul(reshaped_x, w)
        if use_bias:
            b = tf.get_variable('b', [units],
                                initializer=tf.initializers.constant(init_bias))
            out = tf.nn.bias_add(out, b)

        out_shape = tf.concat([in_shape_tensor[:-1], (units,)], axis=0)
        out = tf.reshape(out, out_shape)

        return out
