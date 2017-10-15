# -*- coding:utf-8 -*-

"""
64000=(64*1000)点に対する勾配を求めるまでの時間を計測する。
warmupとして100stepとる
"""

import re
import argparse
import sys
import time

import numpy as np
import tensorflow as tf

from vgg16 import Network
from mnist import MNIST
from tensorflow.python.framework import ops


SEED = 6438
NUM_LABELS = 10
TOTAL_DATA = 8000
BATCH_SIZE = 64
NUM_EPOCHS = 10

FLAGS = None

def overwrite_conv2d_grad():
    gradient_list = ops._gradient_registry.list()

    for grad in gradient_list:
        if "Conv" in grad:
            print (grad)

    del ops._gradient_registry._registry["Conv2D"]

    @ops.RegisterGradient("Conv2D")
    def _conv2d_point_backprop_filter(op, out_grad):
        input = op.inputs[0]
        N, H, W, in_ch = input.shape.as_list()
        _, _, _, out_ch = out_grad.shape.as_list()

        # [NHWC] => [CHWN]
        # SAME用padding
        depth_input = tf.pad(input, [[0,0],[1,1],[1,1],[0,0]],"CONSTANT")
        depth_input = tf.transpose(depth_input, [3, 1, 2, 0])
        #depth_input = tf.transpose(input, [3, 1, 2, 0])
        # [NHWC] => [HWNC]
        depth_out_grad = tf.transpose(out_grad, [1, 2, 0, 3])

        # [in_channel,H,W,N]に[H,W,N,out_channel]をdepthwise_convして、
        # [in_channel,H,W,Nxout_channel]を出力する
        per_depth = tf.nn.depthwise_conv2d(depth_input,
                                           depth_out_grad, strides=[1, 1, 1, 1], padding="VALID",
                                           name="per_depth")
        filter_H, filter_W = per_depth.shape.as_list()[1:3]

        per_depth = tf.reshape(per_depth, [in_ch, filter_H, filter_W, -1, out_ch])
        # [in_channel,H,W,N,out_channel] => [N,H,W,in_channel,out_channel]
        per_point_grad = tf.transpose(per_depth, [3, 1, 2, 0, 4], name="per_point_grad")
        grad_filter = tf.reduce_sum(per_point_grad, axis=0)

        grad_input = tf.nn.conv2d_backprop_input(input.shape, op.inputs[1], out_grad, strides=[1,1,1,1],padding="SAME")

        return grad_input, grad_filter

def matching_conv_grad(graph):
    """
    
    :param tf.Graph graph: 
    :return: 
    """
    re_conv = re.compile(r"conv\d/W")
    var_grads = []

    for cur_var in tf.trainable_variables():
        match = re_conv.search(cur_var.op.name)
        if match is not None:
            # パラメーター名 layer1/conv0/Wの勾配は
            #  gradients/layer1/conv0/convolved_grad/per_point_grad:0
            # に該当する。name matching以外で取り出す方法があるか？
            tensor_name = "gradients/" + cur_var.op.name[:-1] + "convolved_grad/per_point_grad:0"
            tensor = graph.get_tensor_by_name(tensor_name)
            print tensor
            var_grads.append((cur_var, tensor))

    return var_grads


def main(_):
    overwrite_conv2d_grad()

    dataset = MNIST(batch_size=BATCH_SIZE, is_train=False)
    images, labels = dataset.dummy_inputs()

    network = Network()

    with tf.device("/gpu:0"):
        logits = network.inference(images)
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits))

    batch = tf.Variable(0, dtype=tf.int32, trainable=False)
    variables = tf.trainable_variables()
    grads = tf.gradients(loss, variables)
    graph = tf.get_default_graph()

    conv_var_grads = matching_conv_grad(graph)

    per_point_grads = []
    for cur_var, cur_grad in conv_var_grads:
        if cur_var.op.name == "layer0/conv0/W":
            first_conv_grad = cur_grad

        per_point_grads.append(cur_grad)


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.summary.FileWriter(logdir="./log", graph=sess.graph)
        print('Initialized!')

        # value check
        """
        cur_images, cur_labels = dataset.next_batch()
        value_grad = sess.run(first_conv_grad, feed_dict={images: cur_images, labels: cur_labels})
        print value_grad[3]
        print value_grad.shape
        sys.exit(0)
        """

        # warmup
        for step in xrange(100):
            cur_images, cur_labels = dataset.next_batch()
            _ = sess.run([grads, per_point_grads], feed_dict={images:cur_images, labels:cur_labels})

        num_iteration = TOTAL_DATA / BATCH_SIZE
        print ("loop {} data batch_size {} iteration {}".format(TOTAL_DATA, BATCH_SIZE, num_iteration))
        start_time = time.time()

        for step in xrange(num_iteration):
            cur_images, cur_labels = dataset.next_batch()
            _ = sess.run([grads, per_point_grads], feed_dict={images: cur_images, labels: cur_labels})

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
