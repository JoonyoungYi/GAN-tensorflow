import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

from configs import *
from model import init_models


def _init_noise(n=BATCH_SIZE):
    return np.random.normal(size=(n, G_INPUT_LAYER_NODE_NUMBER))


def __debug_save_image(session, models, N=10):
    samples = session.run(
        models['G'], feed_dict={models['Z']: _init_noise(N)})

    for idx in range(N):
        sample = samples[idx]
        img = Image.new('RGB', (28, 28))
        img.putdata([(int(i * 255), int(i * 255), int(i * 255))
                     for i in sample])
        img.save('images/{}.png'.format(idx))


def main():
    mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
    models = init_models()

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    batch_number = int(mnist.train.num_examples / BATCH_SIZE)
    D_loss_value, G_loss_value = 0.0, 0.0
    for epoch_idx in range(EPOCH_NUMBER):
        for batch_idx in range(batch_number):
            batch_xs, _ = mnist.train.next_batch(BATCH_SIZE)
            noise = _init_noise()

            for k in range(K):
                _, D_loss_value = session.run(
                    (models['D_train'], models['D_loss']),
                    feed_dict={
                        models['X']: batch_xs,
                        models['Z']: noise,
                    }, )

            _, G_loss_value = session.run(
                (models['G_train'], models['G_loss']),
                feed_dict={
                    models['Z']: noise,
                }, )

        msg = '[EPOCH%5d] ' % epoch_idx
        msg += 'D_loss_value: %.4f, ' % (D_loss_value)
        msg += 'G_loss_value: %.4f' % (G_loss_value)
        print(msg)

        if epoch_idx % 10 == 1:
            __debug_save_image(session, models, N=10)

    print('COMPLETE!')


if __name__ == '__main__':
    main()
