# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = 16 # IMAGE_SIZE * IMAGE_SIZE


def inference(images, hidden1_units):
  levels = 3
  net = images
  prev_size = IMAGE_PIXELS
  for level in xrange(0, levels):
    divider = 2**level
    next_size = IMAGE_PIXELS // divider
    with tf.name_scope('hidden_in_%s' % level):
      weights = tf.Variable(
          tf.truncated_normal([prev_size, next_size],
                              stddev=1.0 / math.sqrt(float(prev_size))),
          name='weights')
      biases = tf.Variable(tf.zeros([next_size]),
                           name='biases')
      net = tf.nn.relu(tf.matmul(net, weights) + biases)
      prev_size = next_size

  for level in xrange(levels - 1, -1, -1):
    divider = 2**level
    next_size = IMAGE_PIXELS // divider
    with tf.name_scope('hidden_out_%s' % level):
      weights = tf.Variable(
          tf.truncated_normal([prev_size, next_size],
                              stddev=1.0 / math.sqrt(float(prev_size))),
          name='weights')
      biases = tf.Variable(tf.zeros([next_size]),
                           name='biases')
      net = tf.nn.relu(tf.matmul(net, weights) + biases)
      prev_size = next_size

  return net

def loss(logits, labels):
  return tf.reduce_mean(tf.squared_difference(logits, labels), name='mse')


def training(loss, learning_rate):
  tf.summary.scalar('loss', loss)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  return loss(logits, labels)
