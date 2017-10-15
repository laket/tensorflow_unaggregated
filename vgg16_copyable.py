#!/usr/bin/python2.7
#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

SEED = 623

class Network(object):
    def __init__(self):
        self.is_first = True
        # key: Variable Value: list of Tensor
        # 内部で保持されるVariableのコピーを保持する
        self.var_copies = {}

    def get_variable(self, shape, initializer, name):
        new_var = tf.get_variable(name, shape=shape, initializer=initializer, dtype=tf.float32)

        copy_var = tf.identity(new_var, name=new_var.op.name+"_copy")

        if self.is_first:
            self.var_copies[new_var] = []

        self.var_copies[new_var].append(copy_var)
        return copy_var

    def get_all_variables(self):
        return self.var_copies

    def inference(self, images):
        with tf.variable_scope("network") as scope:
            if not self.is_first:
                scope.reuse_variables()

            return self._inference(images)

    def _inference(self, images):
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

            with tf.variable_scope("layer{}".format(idx_layer)):

                for idx_conv in range(num_conv):
                    N,H,W,C = featmap.shape.as_list()

                    with tf.variable_scope("conv{}".format(idx_conv)):
                        initializer = tf.contrib.layers.xavier_initializer_conv2d(seed=SEED)

                        """
                        W = tf.Variable(tf.truncated_normal(
                                [filter_size, filter_size, C, channel], stddev=0.01,
                                seed=SEED), name="W")
                        """
                        W = self.get_variable([filter_size, filter_size, C, channel], initializer, name="W")
                        b = self.get_variable([channel], tf.zeros_initializer(), name="b")

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
            with tf.variable_scope("fc{}".format(idx_fc)):
                N, dim = features.shape.as_list()
                channel = 512
                #channel = 4096

                W = self.get_variable([dim, channel], tf.truncated_normal_initializer(stddev=0.01, seed=SEED), name="W")
                b = self.get_variable([channel], tf.zeros_initializer(), name="b")

                features = tf.matmul(features, W) + b
                features = tf.nn.relu(features)

        with tf.variable_scope("fc2"):
            N, dim = features.shape.as_list()
            channel = 10

            W = self.get_variable([dim, channel], tf.truncated_normal_initializer(stddev=0.01, seed=SEED), name="W")
            b = self.get_variable([channel], tf.zeros_initializer(), name="b")

            logits = tf.matmul(features, W) + b

        self.is_first = False
        return logits
