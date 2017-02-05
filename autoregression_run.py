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

import argparse
import os.path
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import autoregression
import math

FLAGS = None

def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, autoregression.NR_OF_FEATURES))
  labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, autoregression.NR_OF_FEATURES))
  return images_placeholder, labels_placeholder


batchnr = 0
def fill_feed_dict(dataset, images_pl, labels_pl):
  global batchnr
  start = (batchnr * FLAGS.batch_size) % len(dataset)
  stop = start + FLAGS.batch_size
  data = dataset[start:stop,:]
  batchnr += 1

  for col in xrange(0, data.shape[1]):
    data[:,col] = data[:,col] - dataset[:,col].mean()
    data[:,col] = data[:,col] / dataset[:,col].std()

  feed_dict = {
      images_pl: data,
      labels_pl: data,
  }
  return feed_dict


def do_eval(sess,
            loss,
            images_placeholder,
            labels_placeholder,
            data_set):
  steps_per_epoch = 100 # data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  err = 0
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    err += sess.run(loss, feed_dict=feed_dict)
  err = float(err) / steps_per_epoch
  print('  Num examples: %d  Loss: %0.04f' %
        (num_examples, err))
  return err

def run_training():
  dataset = np.load("act_train.npz")['x'][:1000]
  dataset = dataset.view((np.float32, len(dataset.dtype.names)))[:,1:]

  with tf.Graph().as_default():
    np.random.seed(1234)
    tf.set_random_seed(12)
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    logits = autoregression.inference(images_placeholder, FLAGS.divider, FLAGS.levels, FLAGS.level_thickness)
    loss = autoregression.loss(logits, labels_placeholder)
    autoregression.evaluate_absolute_error(logits, labels_placeholder)
    train_op = autoregression.training(loss, FLAGS.learning_rate)
    eval_correct = autoregression.evaluation(logits, labels_placeholder)
    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    sess.run(init)
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      feed_dict = fill_feed_dict(dataset,
                                 images_placeholder,
                                 labels_placeholder)
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      if step % 100 == 0:
        print('Step %d: loss = %0.04f (%.3f sec)' % (step, loss_value, duration))

        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        print('Training Data Eval:')
        do_eval(sess,
                loss,
                images_placeholder,
                labels_placeholder,
                dataset)

    return do_eval(sess,
                   loss,
                   images_placeholder,
                   labels_placeholder,
                   dataset)

def run_cross():
    res = {}
    for lre in xrange(-5, 2):
        lr = 2.0**lre
        FLAGS.learning_rate = lr
        res[lr] = run_training()
        print("LEARNING RATE %s: loss=%s" % (lr, res[lr]))
    print()
    print("Larning rate: Loss")
    for key, value in res.iteritems():
        print("%s: %s" % (key, value))

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()

def cross_main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_cross()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--cross_validate',
      default=False,
      help='Cross validate for learning rates.',
      action='store_true'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.25,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--levels',
      type=int,
      default=1,
      help='Number of levels to half the layer width.'
  )
  parser.add_argument(
      '--level_thickness',
      type=int,
      default=1,
      help='Number of layers of the same width'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=30000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--divider',
      type=float,
      default=2.,
      help='Divide nr of nodes by this for each level.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='/tmp/tensorflow/autoregression/input_data',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/tmp/tensorflow/autoregression/logs/fully_connected_feed',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()

  if FLAGS.cross_validate:
    tf.app.run(main=cross_main, argv=[sys.argv[0]] + unparsed)
  else:
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
