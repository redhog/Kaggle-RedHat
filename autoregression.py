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


def inference(data_placeholder, levels=3, level_thickness=1):
  net = data_placeholder
  prev_size = IMAGE_PIXELS
  for level in xrange(0, levels):
    divider = 2**level
    next_size = IMAGE_PIXELS // divider
    for layer in xrange(level_thickness):
      with tf.name_scope('hidden_in_%s_%s' % (level, layer)):
        weights = tf.Variable(
          tf.truncated_normal([prev_size, next_size],
                              stddev=1.0 / math.sqrt(float(prev_size))),
          name='weights')
        biases = tf.Variable(tf.zeros([next_size]),
                             name='biases')
        net = tf.nn.relu(tf.matmul(net, weights) + biases)
        prev_size = next_size
        tf.summary.histogram('weights_in_%s_%s' % (level, layer), weights)
        tf.summary.histogram('biases_in_%s_%s' % (level, layer), biases)

  for level in xrange(levels - 1, -1, -1):
    divider = 2**level
    next_size = IMAGE_PIXELS // divider
    for layer in xrange(level_thickness):
      with tf.name_scope('hidden_out_%s_%s' % (level, layer)):
        weights = tf.Variable(
          tf.truncated_normal([prev_size, next_size],
                              stddev=1.0 / math.sqrt(float(prev_size))),
          name='weights')
        biases = tf.Variable(tf.zeros([next_size]),
                             name='biases')
        net = tf.nn.relu(tf.matmul(net, weights) + biases)
        prev_size = next_size
        tf.summary.histogram('weights_out_%s_%s' % (level, layer), weights)
        tf.summary.histogram('biases_out_%s_%s' % (level, layer), biases)

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

def evaluate_absolute_error(logits, labels):
  tf.summary.histogram('absolute error', logits-labels)
  return logits - labels
