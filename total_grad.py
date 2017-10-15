# -*- coding:utf-8 -*-

"""
64000=(64*1000)点に対する勾配を求めるまでの時間を計測する。
warmupとして100stepとる
"""

import argparse
import sys
import time

import numpy as np
import tensorflow as tf

from vgg16 import Network
from mnist import MNIST

SEED = 6438
NUM_LABELS = 10
TOTAL_DATA = 16000
BATCH_SIZE = 1
NUM_EPOCHS = 10

FLAGS = None


def main(_):
    dataset = MNIST(batch_size=BATCH_SIZE, is_train=False)
    images, labels = dataset.dummy_inputs()

    network = Network()

    with tf.device("/gpu:0"):
        logits = network.inference(images)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits))

    batch = tf.Variable(0, dtype=tf.int32, trainable=False)
    variables = tf.trainable_variables()
    grads = tf.gradients(loss, variables)

    #grad_vars = []
    valid_grads = []
    for cur_var, cur_grad in zip(variables, grads):
        if cur_grad is None:
            print "{} has no grad".format(cur_var.op.name)
        else:
            valid_grads.append(cur_grad)

        if cur_var.op.name == "layer0/conv0/W":
            first_conv_grad = cur_grad

    grads = valid_grads

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.summary.FileWriter(logdir="./log", graph=sess.graph)
        print('Initialized!')

        # value check
        """
        cur_images, cur_labels = dataset.next_batch()
        value_grad = sess.run(first_conv_grad, feed_dict={images: cur_images, labels: cur_labels})
        print value_grad
        print value_grad.shape
        sys.exit(0)
        """

        # warmup
        for step in xrange(100):
            cur_images, cur_labels = dataset.next_batch()
            _ = sess.run(grads, feed_dict={images:cur_images, labels:cur_labels})

        num_iteration = TOTAL_DATA / BATCH_SIZE
        print ("loop {} data batch_size {} iteration {}".format(TOTAL_DATA, BATCH_SIZE, num_iteration))
        start_time = time.time()

        for step in xrange(num_iteration):
            cur_images, cur_labels = dataset.next_batch()
            _ = sess.run(grads, feed_dict={images: cur_images, labels: cur_labels})

        duration = time.time() - start_time
        num_data = num_iteration * BATCH_SIZE
        print ("duraction {} time/data {}msec".format(duration, duration*1000/num_data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--use_fp16',
            default=False,
            help='Use half floats instead of full floats if True.',
            action='store_true')
    parser.add_argument(
            '--self_test',
            default=False,
            action='store_true',
            help='True if running a self test.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
