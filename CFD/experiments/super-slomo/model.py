
import tensorflow as tf


def _u_net(x):
    # Encoder
    num_feats = (x.shape[3] // 2).value

    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Decoder
    convt6 = tf.keras.layers.Conv2DTranspose(512, 2 * 2, strides=2, activation='relu', padding='same')(conv5)
    merge6 = tf.concat([conv4, convt6], axis=3)

    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    convt7 = tf.keras.layers.Conv2DTranspose(256, 2 * 2, strides=2, activation='relu', padding='same')(conv6)
    merge7 = tf.concat([conv3, convt7], axis=3)

    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    convt8 = tf.keras.layers.Conv2DTranspose(128, 2 * 2, strides=2, activation='relu', padding='same')(conv7)
    merge8 = tf.concat([conv2, convt8], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    convt9 = tf.keras.layers.Conv2DTranspose(64, 2 * 2, strides=2, activation='relu', padding='same')(conv8)
    merge9 = tf.concat([conv1, convt9], axis=3)

    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = tf.keras.layers.Conv2D(num_feats, 2, activation='relu', padding='same')(conv9)

    conv10 = tf.keras.layers.Conv2D(num_feats, 2, padding='same')(conv9)

    return conv9

def build_model(x0, x1):
    x = tf.concat([tf.expand_dims(x0, 3), tf.expand_dims(x1, 3)], 4)
    n, h, w, d, t = x.shape

    # Perform instance normalization
    # mean = tf.stop_gradient(tf.reshape(tf.math.reduce_mean(x, axis=(1, 2, 3)), [n, 1, 1, 1, d]))
    # std = tf.stop_gradient(tf.reshape(tf.math.reduce_std(x, axis=(1, 2, 3)), [n, 1, 1, 1, d]))
    # x = x - mean / std

    # Collapse t, d into single dimension
    x = tf.reshape(x, [n, h, w, t * d])

    # Pass model through unet
    y = _u_net(x)

    # Restore normalization on reconstruction
    # mean = tf.reshape(mean, [n, 1, 1, d])
    # std = tf.reshape(std, [n, 1, 1, d])
    # y = y * std + mean
    return y

if __name__ == "__main__":
    tf.enable_eager_execution()
    
    x = tf.random.uniform(shape=[1, 128, 128, 4])
    y = build_model(x, x)
    print(y.shape)