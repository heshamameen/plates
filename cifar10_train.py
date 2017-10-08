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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import subprocess

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/output',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 1,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_string('checkpoint_dir',
                           '/ckpts/model.ckpt-4000',
                           """Directory where to read model checkpoints.""")


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        # global_step_init = -1
        # global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images)

        # variable_averages = tf.train.ExponentialMovingAverage(
        #     cifar10.MOVING_AVERAGE_DECAY)
        # variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver()
        # Calculate loss.
        loss = cifar10.loss(logits, labels)

        # if ckpt and ckpt.model_checkpoint_path:
        #     global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])


        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        global_step = tf.contrib.framework.get_or_create_global_step()
        train_op = cifar10.train(loss, global_step)



        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()
                # cmd_line = ('tensorboard --log %s' % FLAGS.train_dir)
                # p = subprocess.Popen(cmd_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                         examples_per_sec, sec_per_batch))

        # with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.train_dir,
        #                                        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps), tf.train.NanTensorHook(loss), _LoggerHook()],
        #                                        config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement), save_checkpoint_secs=3600,
        #                                        save_summaries_steps=5) as mon_sess:
        saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.train_dir, save_steps=500,
                                                  saver=tf.train.Saver(max_to_keep=20))
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook(), saver_hook],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement),save_checkpoint_secs=9999999,save_summaries_steps=10) as mon_sess:
            # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            # if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                # ckpt_dir = ckpt.model_checkpoint_path
                # ckpt_dir = ckpt_dir.split("/")
                # ckpt_dir = ckpt_dir[-1]

            # ckpt_dir = FLAGS.checkpoint_dir 
            # saver.restore(mon_sess, ckpt_dir)
            
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            # coord = tf.train.Coordinator()
            # try:
            #   threads = []
            #   for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            #     threads.extend(qr.create_threads(mon_sess, coord=coord, daemon=True,
            #                                      start=True))
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
            # except Exception as e:  # pylint: disable=broad-except
            #   coord.request_stop(e)
            #
            # coord.request_stop()
            # coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):  # pylint: disable=unused-argument
    # cifar10.maybe_download_and_extract()
    # if tf.gfile.Exists(FLAGS.train_dir):
        # tf.gfile.DeleteRecursively(FLAGS.train_dir)
    # tf.gfile.MakeDirs(FLAGS.train_dir)
    # out = p.communicate()[0]
    # print (out)
    train()


if __name__ == '__main__':
    tf.app.run()
