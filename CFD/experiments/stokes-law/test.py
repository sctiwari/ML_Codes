
''' Navier stokes law for fluid flow written in TensorFlow '''

import tensorflow as tf

tf.enable_eager_execution()

def grad_3d(flow, grad_dir, scale=1.0):
    ''' Computes the numeric 3D gradients along x, y, z
        flow: 5D tensor of shape NDHWC with C = 1
        grad_dir: direction along wich gradient must be computed
        scale: float specifying
    '''
    assert flow.shape[5] == 1

    scale = tf.constant(scale)
    if grad_dir == 'x':
        kernel = tf.reshape(tf.constant([-1.0, 0.0, 1.0]), [1, 1, 3, 1, 1])
    elif grad_dir == 'y':
        kernel = tf.reshape(tf.constant([-1.0, 0.0, 1.0]), [1, 1, 1, 3, 1])
    elif grad_dir == 'z':
        kernel = tf.reshape(tf.constant([-1.0, 0.0, 1.0]), [1, 1, 1, 1, 3])
    else:
        raise NotImplementedError('unknown direction')
    return tf.math.divide(tf.nn.conv3d(flow, kernel, [1, 1, 1, 1, 1], 'SAME'), scale)

def grad_time(flow_0, flow_1, scale=1.0):
    return tf.math.divide(flow_1 - flow_0, scale)

if __name__ == '__main__':

    # TODO load some file
    flow = None
    flow_next = None

    delta = tf.reduce_sum(
        grad_3d(flow, 'x') + grad_3d(flow, 'y') + grad_3d(flow, 'z') + grad_time(flow, flow_next))
    
    print(delta)
