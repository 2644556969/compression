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
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import math
import sys
import time

import numpy as np
import tensorflow as tf

import models.data as data
import models.select as select
import os
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.tools import freeze_graph
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import rewriter_config_pb2


import ngraph_bridge

from train import get_run_dir

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer('model_number', -1,
                            """Specified model number in a directory""")
tf.app.flags.DEFINE_string('backend', 'HE_SEAL', "backend")
tf.app.flags.DEFINE_string('encryption_parameters', '', "Filename containing json description of encryption parameters, or json description itself")
tf.app.flags.DEFINE_boolean('encrypt_server_data', False,
                            """Encrypt server data (should not be used when enable_client is used)""")

tf.app.flags.DEFINE_boolean('resume', False,
                            """Continue training the previous model""")




#SET UP CONFIGS TO USE HE-TRANSFORMER BACKEND (from MNIST_UTIL)

def server_config_from_flags(FLAGS, tensor_param_name):
    rewriter_options = rewriter_config_pb2.RewriterConfig()
    rewriter_options.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE
    rewriter_options.min_graph_nodes = -1
    server_config = rewriter_options.custom_optimizers.add()
    server_config.name = "ngraph-optimizer"
    server_config.parameter_map["ngraph_backend"].s = FLAGS.backend.encode()
    server_config.parameter_map["device_id"].s = b""
    server_config.parameter_map[
        "encryption_parameters"].s = FLAGS.encryption_parameters.encode()
    server_config.parameter_map["enable_client"].s = str(
        False).encode()
    server_config.parameter_map["enable_gc"].s = (str(False)).encode()
    server_config.parameter_map["mask_gc_inputs"].s = (str(
        False)).encode()
    server_config.parameter_map["mask_gc_outputs"].s = (str(
        False)).encode()
    server_config.parameter_map["num_gc_threads"].s = (str(
        False)).encode()


    if FLAGS.encrypt_server_data:
        server_config.parameter_map[tensor_param_name].s = b"encrypt"

    config = tf.compat.v1.ConfigProto()
    config.MergeFrom(
        tf.compat.v1.ConfigProto(
            graph_options=tf.compat.v1.GraphOptions(
                rewrite_options=rewriter_options)))

    return config



def save_weights():
    """Saves CIFAR10 weights"""
    FLAGS.resume = True  # Get saved weights, not new ones
    print(FLAGS.save_dir)
    run_dir = get_run_dir(FLAGS.save_dir, FLAGS.model, model_number = FLAGS.model_number)
    print('run_dir', run_dir)
    checkpoint_dir = os.path.join(run_dir, 'train')

    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        images, labels = data.train_inputs(data_dir=FLAGS.data_dir)
        model = select.by_name(FLAGS.model, FLAGS, training=True)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(images)
        print('Multiplicative depth', model.mult_depth())

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split(
                    '-')[-1]
            else:
                print('### ERROR No checkpoint file found###')
                print('ckpt_dir', checkpoint_dir)
                print('ckpt.model_checkpoint_path', ckpt.model_checkpoint_path)
                print('ckpt', ckpt)
                return

            # Save variables
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                weight = (sess.run([var]))[0].flatten().tolist()
                filename = model._name_to_filename(var.name)
                dir_name = filename.rsplit('/', 1)[0]
                os.makedirs(dir_name, exist_ok=True)

                print("saving", filename)
                np.savetxt(str(filename), weight)


def optimize_model_for_inference():
    """Optimizes CIFAR-10 model for inference"""
    FLAGS.resume = True  # Get saved weights, not new ones
    run_dir = get_run_dir(FLAGS.save_dir, FLAGS.model, model_number = FLAGS.model_number)
    checkpoint_dir = os.path.join(run_dir, 'train')
    print('run_dir', run_dir)
    print('checkpoint dir', checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    train_graph = os.path.join(checkpoint_dir, 'graph.pbtxt')
    frozen_graph = os.path.join(checkpoint_dir, 'graph_constants.pb')
    fused_graph = os.path.join(checkpoint_dir, 'fused_graph.pb')

    with tf.Session() as sess:
        # TODO this should be a placeholder, right?
        # Build a new inference graph, with variables to be restored from
        # training graph.
        IMAGE_SIZE = 24 if FLAGS.data_aug else 32
        if FLAGS.batch_norm:
            images = tf.constant(
                1, dtype=tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, 3])
        else:
            images = tf.constant(
                1,
                dtype=tf.float32,
                shape=[FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])

        model = select.by_name(FLAGS.model, FLAGS, training=False)
        # Create dummy input and output nodes
        images = tf.identity(images, 'XXX')
        logits = model.inference(images)
        logits = tf.identity(logits, 'YYY')

        if FLAGS.batch_norm:
            # Restore values from the trained model into corresponding variables in the
            # inference graph.
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            print('ckpt.model_checkpoint_path', ckpt.model_checkpoint_path)
            assert ckpt and ckpt.model_checkpoint_path, "No checkpoint found in {}".format(
                checkpoint_dir)

            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)

            # Write fully-assembled inference graph to a file, so freeze_graph can use it
            tf.io.write_graph(
                sess.graph,
                checkpoint_dir,
                'inference_graph.pbtxt',
                as_text=True)

            # Freeze graph, converting variables to inline-constants in pb file
            constant_graph = os.path.join(checkpoint_dir, 'graph_constants.pb')
            freeze_graph.freeze_graph(
                input_graph=os.path.join(checkpoint_dir,
                                         'inference_graph.pbtxt'),
                input_saver="",
                input_binary=False,
                input_checkpoint=ckpt.model_checkpoint_path,
                output_node_names='YYY',
                restore_op_name='save/restore_all',
                filename_tensor_name='save/Const:0',
                initializer_nodes=[],
                output_graph=os.path.join(checkpoint_dir,
                                          'graph_constants.pb'),
                clear_devices=True)

            # Load frozen graph into a graph_def for optimize_lib to use
            with gfile.FastGFile(constant_graph, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')

            # Optimize graph for inference, folding Batch Norm ops into conv/MM
            fused_graph_def = optimize_for_inference_lib.optimize_for_inference(
                input_graph_def=graph_def,
                input_node_names=['XXX'],
                output_node_names=['YYY'],
                placeholder_type_enum=dtypes.float32.as_datatype_enum,
                toco_compatible=False)

            print('Optimized for inference.')

            tf.io.write_graph(
                fused_graph_def,
                checkpoint_dir,
                name='fused_graph.pb',
                as_text=False)
        else:
            tf.io.write_graph(
                sess.graph, checkpoint_dir, 'fused_graph.pb', as_text=False)


def report_accuracy(logits, labels):
    correct_prediction = np.equal(np.argmax(logits, 1), labels)
    error_count = np.size(correct_prediction) - np.sum(correct_prediction)
    test_accuracy = np.mean(correct_prediction)

    print('Error count', error_count, 'of', len(labels), 'elements.')
    print('Accuracy ', test_accuracy)


def perform_inference():
    print('Performing inference')

    FLAGS.resume = True  # Get saved weights, not new ones
    run_dir = get_run_dir(FLAGS.save_dir, FLAGS.model, model_number = FLAGS.model_number)
    checkpoint_dir = os.path.join(run_dir, 'train')
    fused_graph_file = os.path.join(checkpoint_dir, 'fused_graph.pb')

    eval_data, eval_labels = data.numpy_eval_inputs(True, FLAGS.data_dir,
                                                    FLAGS.batch_size)

    with gfile.FastGFile(fused_graph_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        assert len(graph.get_operations(
        )) == 0, "Assuming an empty graph here to populate with fused graph"
        tf.import_graph_def(graph_def, name='')

    print('nodes', [n.name for n in graph_def.node])
    XXX = graph.get_tensor_by_name('XXX:0')
    YYY = graph.get_tensor_by_name('YYY:0')

    print("GETTING CONFIGS")

    config = server_config_from_flags(FLAGS, XXX.name)

    print("Running model")
    with tf.Session(graph=graph, config=config) as sess:
        start_time = time.time()
        eval_batch_data = eval_data[0]
        eval_batch_label = eval_labels[0]
        YYY = sess.run(YYY, feed_dict={XXX: eval_batch_data})
        elapsed_time = time.time() - start_time 
        print("total time(s)", np.round(elapsed_time, 6))

    report_accuracy(YYY, eval_batch_label)


def main(argv=None):
    data.maybe_download_and_extract(FLAGS.data_dir)

    save_weights()
    optimize_model_for_inference()
    perform_inference()


if __name__ == '__main__':
    tf.app.run()
