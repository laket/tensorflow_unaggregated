# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import gzip
import logging

import numpy as np
import cv2

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

FLAGS = tf.app.flags.FLAGS
DIR_MNIST = "./data"
# CVDF mirror of http://yann.lecun.com/exdb/mnist/
SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'


class MNIST(object):
    def __init__(self, is_train, batch_size=64, dir_root=DIR_MNIST):
        """
        In train mode:
          inlier only
          data is shuffled
          Each data is used many times.
        In test mode:
          inlier and outlier
          data is NOT shuffled
          Each data is used one time.

        :param boolean is_train: train mode or not
        :param int batch_size: batch size
        """
        self.is_train = is_train
        self.batch_size = batch_size
        self._completed = False

        self._images, self._labels = self._load(dir_root)

        self._num_data = len(self._images)
        self._reset_indices()

    def _load(self, dir_root):
        """
        ディレクトリから画像を読み込む
        :param dir_root: 
        :return:
          (images, labels)
          label == 0 : indicates INLIER
          label == 1 : indicates OUTLIER
        """
        logger = logging.getLogger(__name__)
        train_images, train_labels, test_images, test_labels = download_mnist(DIR_MNIST)

        if self.is_train:
            return np.array(train_images), np.array(train_labels)
        else:
            return np.array(test_images), np.array(test_labels)

    def _reset_indices(self):
        self._indices = range(self._num_data)
        self._idx_next = 0

        if self.is_train:
            np.random.shuffle(self._indices)

        return self._images

    def _next_indices(self, batch_size):
        self._completed = False
        indices = []

        # epochをまたぐケース
        if self._idx_next + batch_size >= self._num_data:
            indices += self._indices[self._idx_next:self._num_data]
            batch_size -= self._num_data - self._idx_next
            self._reset_indices()
            self._completed = True

        indices += self._indices[self._idx_next:self._idx_next + batch_size]
        self._idx_next += batch_size

        return indices

    @property
    def completed(self):
        """
        In test mode:
          すべてのデータを利用済みかどうか
        In train mode:
          直前のnext_batchでepochをまたいだかどうか
        :return: 
        """
        return self._completed

    def preprocess(self, images):
        """
        preprocess images (commonly train mode and test mode)
        :param np.ndarray images: uint8 [NHWC]
        :return: 
        """
        images = images.astype(dtype=np.float32)

        return images / 255

    def depreprocess(self, images):
        """
        reverse preprocessed images to original images (commonly train mode and test mode)
        :param np.ndarray images: float32 [NHWC]
        :return: 
        """
        images = np.array(images)
        images = images * 255
        images = images.astype(dtype=np.uint8)
        return images

    def next_batch(self):
        """
        :rtype: (np.ndarray, np.ndarray)
        :return: 
          (images, labels)
          images: preprocessed batch images (float32 NCHW)
          labels: batch labels (uint8 [N])
        """
        indices = self._next_indices(self.batch_size)

        images = self._images[indices]
        labels = self._labels[indices]

        images, labels = np.stack(images), np.stack(labels)
        images = self.preprocess(images)

        return images, labels

    def dummy_inputs(self):
        images = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 28, 28, 1])
        labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])

        return images, labels



def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 np array [index, y, x, depth].
    Args:
        f: A file object that can be passed into a gzip reader.
    Returns:
        data: A 4D uint8 np array [index, y, x, depth].
    Raises:
        ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 np array [index].
    Args:
        f: A file object that can be passed into a gzip reader.
        one_hot: Does one hot encoding for the result.
        num_classes: Number of classes for the one hot encoding.
    Returns:
        labels: a 1D uint8 np array.
    Raises:
        ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
        return labels

def download_mnist(train_dir):
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                     SOURCE_URL + TRAIN_IMAGES)
    with open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                     SOURCE_URL + TRAIN_LABELS)
    with open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=False)

    local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                     SOURCE_URL + TEST_IMAGES)
    with open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = base.maybe_download(TEST_LABELS, train_dir,
                                     SOURCE_URL + TEST_LABELS)
    with open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=False)

    return train_images, train_labels, test_images, test_labels


