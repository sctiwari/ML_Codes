
import os

import tensorflow as tf

from model import build_model
from dataset import iterate_dataset, iterate_sequences, shuffle

eps = 1e-10

if __name__ == '__main__':

    batch_size = 1
    res = 256
    num_features = 1

    x0 = tf.placeholder(tf.float32, shape=[batch_size, res, res, num_features])
    x1 = tf.placeholder(tf.float32, shape=[batch_size, res, res, num_features])
    y = tf.placeholder(tf.float32, shape=[batch_size, res, res, num_features])

    y_pred = build_model(x0, x1)

    loss = tf.losses.absolute_difference(y, y_pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    mean_abs_err = tf.reduce_mean(tf.abs(y_pred - y))
    mean_sqr_err = tf.reduce_mean(tf.math.square(y_pred - y))

    summaries = tf.summary.merge([
        tf.summary.scalar('loss', loss),
        tf.summary.scalar('mean_abs_err', mean_abs_err),
        tf.summary.scalar('mean_sqr_err', mean_sqr_err)
    ])

    import datetime
    timenow = datetime.datetime.now()
    writer_train = tf.summary.FileWriter(f'train_out/{timenow}/tensorboard/train',
                                         tf.get_default_graph())
    writer_valid = tf.summary.FileWriter(f'train_out/{timenow}/tensorboard/valid',
                                         tf.get_default_graph())
    os.makedirs(f'train_out/{timenow}/checkpoints')

    saver = tf.train.Saver()

    sess_names = [
        "2019-03-24_16-11-50_res_128",
        "2019-03-24_17-14-36_res_128",
        "2019-03-24_18-39-29_res_128",
        "2019-03-24_19-21-11_res_128",
        "2019-03-24_16-14-01_res_128",
        "2019-03-24_17-16-38_res_128",
        "2019-03-24_18-39-39_res_128",
        "2019-03-24_19-22-01_res_128",
        "2019-03-24_16-15-22_res_128",
        "2019-03-24_17-20-35_res_128",
        "2019-03-24_18-40-43_res_128",
        "2019-03-24_19-23-14_res_128",
        "2019-03-24_16-16-34_res_128",
        "2019-03-24_17-22-39_res_128",
        "2019-03-24_18-40-51_res_128",
        "2019-03-24_19-24-49_res_128",
        "2019-03-24_16-18-50_res_128",
        "2019-03-24_17-24-23_res_128",
        "2019-03-24_18-43-09_res_128",
        "2019-03-24_19-25-32_res_128",
        "2019-03-24_16-20-28_res_128",
        "2019-03-24_17-28-31_res_128",
        "2019-03-24_18-45-31_res_128",
        "2019-03-24_19-26-49_res_128",
        "2019-03-24_16-22-04_res_128",
        "2019-03-24_17-31-12_res_128",
        "2019-03-24_18-46-58_res_128",
        "2019-03-24_19-28-03_res_128",
        "2019-03-24_16-23-34_res_128",
        "2019-03-24_17-33-00_res_128",
        "2019-03-24_18-48-45_res_128",
        "2019-03-24_19-30-00_res_128",
        "2019-03-24_16-25-43_res_128",
        "2019-03-24_17-36-06_res_128",
        "2019-03-24_18-50-30_res_128",
        "2019-03-24_19-31-45_res_128",
        "2019-03-24_16-27-02_res_128",
        "2019-03-24_17-37-57_res_128",
        "2019-03-24_18-52-06_res_128",
        "2019-03-24_19-33-07_res_128",
        "2019-03-24_16-28-14_res_128",
        "2019-03-24_17-41-50_res_128",
        "2019-03-24_18-52-24_res_128",
        "2019-03-24_19-34-28_res_128",
        "2019-03-24_16-30-11_res_128",
        "2019-03-24_17-45-04_res_128",
        "2019-03-24_18-53-58_res_128",
        "2019-03-24_19-35-13_res_128",
        "2019-03-24_16-33-28_res_128",
        "2019-03-24_17-48-38_res_128",
        "2019-03-24_18-55-41_res_128",
        "2019-03-24_19-36-14_res_128",
        "2019-03-24_16-37-03_res_128",
        "2019-03-24_17-50-53_res_128",
        "2019-03-24_18-56-57_res_128",
        "2019-03-24_19-36-24_res_128",
        "2019-03-24_16-41-09_res_128",
        "2019-03-24_17-54-01_res_128",
        "2019-03-24_18-57-59_res_128",
        "2019-03-24_19-38-07_res_128",
        "2019-03-24_16-45-08_res_128",
        "2019-03-24_17-56-30_res_128",
        "2019-03-24_18-59-58_res_128",
        "2019-03-24_19-39-08_res_128",
        "2019-03-24_16-47-35_res_128",
        "2019-03-24_17-59-25_res_128",
        "2019-03-24_19-01-56_res_128",
        "2019-03-24_19-40-35_res_128",
        "2019-03-24_16-49-54_res_128",
        "2019-03-24_18-03-15_res_128",
        "2019-03-24_19-03-59_res_128",
        "2019-03-24_19-40-45_res_128",
        "2019-03-24_16-51-55_res_128",
        "2019-03-24_18-07-29_res_128",
        "2019-03-24_19-05-22_res_128",
    ]

    train_sessions = sess_names[:50]
    valid_sessions = sess_names[50:]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        prev_sess = None

        for epoch in range(5000):

            for data in shuffle(iterate_sequences(iterate_dataset('data-good'), window_size=1)):

                x0_val = data[0]['density']
                x1_val = data[2]['density']
                y_val = data[1]['density']

                if sess_names in train_sessions:
                    _, summary = sess.run(
                        [optimizer, summaries], feed_dict={
                            x0: x0_val,
                            x1: x1_val,
                            y: y_val,
                        })

                    writer_train.add_summary(summary, epoch)
                else:
                    summary = sess.run(
                        summaries, feed_dict={
                            x0: x0_val,
                            x1: x1_val,
                            y: y_val,
                        })

                    writer_valid.add_summary(summary, epoch)

            saver.save(sess, f'train_out/{timenow}/checkpoints/latest.ckpt')
