#!/usr/bin/python2.7
#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

SEED = 623

class Network(object):
    def __init__(self):
        pass

    def inference(self, images):
        # layers[i] = (filter_size, channel, num_conv)
        conv_layers = [
            (3,  64, 2),
            (3, 128, 2),
            (3, 256, 3),
#            (3, 512, 3),
#            (3, 512, 3)
        ]

        #images = tf.image.resize_images(images, size=(224,224))
        images = tf.image.resize_images(images, size=(64,64))

        featmap = images

        for idx_layer, cur_layer in enumerate(conv_layers):
            filter_size, channel, num_conv = cur_layer

            with tf.name_scope("layer{}".format(idx_layer)) as scope:

                for idx_conv in range(num_conv):
                    N,H,W,C = featmap.shape.as_list()

                    with tf.name_scope("conv{}".format(idx_conv)):
                        initializer = tf.contrib.layers.xavier_initializer_conv2d(seed=SEED)

                        """
                        W = tf.Variable(tf.truncated_normal(
                                [filter_size, filter_size, C, channel], stddev=0.01,
                                seed=SEED), name="W")
                        """
                        W = tf.Variable(initializer([filter_size, filter_size, C, channel]), name="W")

                        b = tf.Variable(np.zeros(shape=[channel]), dtype=tf.float32, name="b")


                        featmap = tf.nn.conv2d(
                                featmap,
                                W, strides=[1, 1, 1, 1], padding='SAME', name="convolved")

                        featmap = tf.nn.relu(tf.nn.bias_add(featmap, b))


                featmap = tf.nn.max_pool(featmap,
                                         ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')

        N,H,W,C = featmap.shape.as_list()
        features = tf.reshape(featmap, [-1, H*W*C])

        for idx_fc in range(2):
            with tf.name_scope("fc{}".format(idx_fc)):
                N, dim = features.shape.as_list()
                channel = 512
                #channel = 4096

                W = tf.Variable(tf.truncated_normal([dim, channel], stddev=0.01, seed=SEED), name="W")
                b = tf.Variable(np.zeros(shape=[channel]), dtype=tf.float32, name="b")


                features = tf.matmul(features, W) + b
                features = tf.nn.relu(features)

        with tf.name_scope("fc2"):
            N, dim = features.shape.as_list()
            channel = 10

            W = tf.Variable(tf.truncated_normal([dim, channel], stddev=0.01, seed=SEED), name="W")
            b = tf.Variable(np.zeros(shape=[channel]), dtype=tf.float32, name="b")

            logits = tf.matmul(features, W) + b

        return logits
