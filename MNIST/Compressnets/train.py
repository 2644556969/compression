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
"""An MNIST classifier based on Cryptonets using convolutional layers. """
"""Supports model compression of training shallow neural networks using logit data"""
"""Generates multiple test architectures for a given cryptographic security level for HE""" 

import sys
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.tools import freeze_graph
import model 

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mnist_util

from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Activation,
    AveragePooling2D,
    Flatten,
    MaxPooling2D,
    Input,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import (SGD, RMSprop, Adam, Nadam)


# Squash linear layers and return squashed weights
def squash_layers(cryptonets_model, sess):
    layers = cryptonets_model.layers
    layer_names = [layer.name for layer in layers]
    conv1_weights = layers[layer_names.index('conv2d_1')].get_weights()
    conv2_weights = layers[layer_names.index('conv2d_2')].get_weights()
    fc1_weights = layers[layer_names.index('fc_1')].get_weights()
    fc2_weights = layers[layer_names.index('fc_2')].get_weights()

    # Get squashed weight
    y = Input(shape=(14 * 14 * 5,), name="squashed_input")
    y = Reshape((14, 14, 5))(y)
    y = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y)
    y = Conv2D(
        filters=50,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_bias=True,
        trainable=False,
        kernel_initializer=tf.compat.v1.constant_initializer(conv2_weights[0]),
        bias_initializer=tf.compat.v1.constant_initializer(conv2_weights[1]),
        name="conv2d_test",
    )(y)
    y = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y)
    y = Flatten()(y)
    y = Dense(
        100,
        use_bias=True,
        name="fc_1",
        kernel_initializer=tf.compat.v1.constant_initializer(fc1_weights[0]),
        bias_initializer=tf.compat.v1.constant_initializer(fc1_weights[1]))(y)

    sess.run(tf.compat.v1.global_variables_initializer())

    # Pass 0 to get bias
    squashed_bias = y.eval(
        session=sess,
        feed_dict={
            "squashed_input:0": np.zeros((1, 14 * 14 * 5))
        })
    squashed_bias_plus_weights = y.eval(
        session=sess, feed_dict={
            "squashed_input:0": np.eye(14 * 14 * 5)
        })
    squashed_weights = squashed_bias_plus_weights - squashed_bias

    print("squashed layers")

    # Sanity check
    x_in = np.random.rand(100, 14 * 14 * 5)
    network_out = y.eval(session=sess, feed_dict={"squashed_input:0": x_in})
    linear_out = x_in.dot(squashed_weights) + squashed_bias
    assert np.max(np.abs(linear_out - network_out)) < 1e-2

    return (conv1_weights, (squashed_weights, squashed_bias), fc1_weights,
            fc2_weights)

# Squash connected linear layers and return squashed weights
#right now doesn't support convolutional layer squashing, only squashing of linear layers
#assume only dense and activation layers in layer list  
def squash_layers_variable(cryptonets_model, sess, layer_list):
    layers = cryptonets_model.layers
    layer_names = [layer.name for layer in layers]

    weights = [] 
    compressed_layer_list = [] 
    curr_input = 784 
    dense_processed = 0 
    dense_added = 0 

    i = 0 
    while i < len(layer_list):
        if layer_list[i][0] == "activation":
            compressed_layer_list.append(layer_list[i]) 
            i += 1 
        else: 
            end = i + 1 
            while end < len(layer_list) and layer_list[end][0] != "activation":
                end += 1 
            #layers to compress are from i to j, excluding j 
            #if only one layer between activation 
            if end - i == 1: 
                name = "dense_" + str(dense_processed)
                layer_info = layers[layer_names.index(name)]
                curr_layer_weights = layer_info.get_weights() 
                weights.append(curr_layer_weights) 
                compressed_layer_list.append(("dense", layer_info.output_shape[1]))
                print("processing layer", layer_info.get_config()) 

                curr_input = layer_info.output_shape[1]
                dense_processed += 1 
                print("added layer", compressed_layer_list) 

            else: 
                #squash dense layers 
                orig_input = curr_input 
                y = Input(shape=(orig_input,), name = "squashed_input")
                for j in range(i, end): 
                    name = "dense_" + str(dense_processed) 
                    layer_info = layers[layer_names.index(name)]
                    print("processing layer", layer_info.get_config()) 
                    layer_weights = layer_info.get_weights() 
                    y = Dense(layer_info.output_shape[1], 
                        use_bias = True, 
                        name=name,  #is this improtant? 
                        kernel_initializer=tf.compat.v1.constant_initializer(layer_weights[0]),
                        bias_initializer=tf.compat.v1.constant_initializer(layer_weights[1]))(y)
                    
                    curr_input = layer_info.output_shape[1] 
                    dense_processed += 1 

                #compress this multi-layer dense model into one layer 

                sess.run(tf.compat.v1.global_variables_initializer())

                # Pass 0 to get bias
                squashed_bias = y.eval(
                    session=sess,
                    feed_dict={
                        "squashed_input:0": np.zeros((1, orig_input))
                    })
                squashed_bias_plus_weights = y.eval(
                    session=sess, feed_dict={
                        "squashed_input:0": np.eye(orig_input)
                    })
                squashed_weights = squashed_bias_plus_weights - squashed_bias
                print("squashed layers")

                #sanity check 
                x_in = np.random.rand(100, orig_input)
                network_out = y.eval(session=sess, feed_dict={"squashed_input:0": x_in})
                linear_out = x_in.dot(squashed_weights) + squashed_bias
                print("max difference: ",  np.max(np.abs(linear_out - network_out)))
                assert np.max(np.abs(linear_out - network_out)) < 1e-2


                #add layer 
                compressed_layer_list.append(("dense", curr_input))
                weights.append((squashed_weights, squashed_bias))
                print("added layer", compressed_layer_list) 

            i = end 

    return weights, compressed_layer_list 

#geenrates a compressed model with pre-trained weights 

def cryptonets_model_no_conv_squashed(input, weights, layer_list):

    #start with a flatten layer and then go thru rest of layer list 

    def square_activation(x):
        return x * x

    def poly_activation(x): 
        return .125 * (x + 2) ** 2 - .25 

    y = tf.reshape(input, [-1, 28 * 28])
    weight_index = 0 
    for i in range(len(layer_list)): 
        if layer_list[i][0] == "dense": 
            if i == len(layer_list) - 1: #last layer 
                name = "output"
            else:
                name = "dense_" + str(weight_index)
            layer_weights = weights[weight_index]
            y = Dense(layer_list[i][1], 
                use_bias=True, 
                name = name, 
                trainable = False, 
                kernel_initializer=tf.compat.v1.constant_initializer(layer_weights[0]),
                bias_initializer=tf.compat.v1.constant_initializer(layer_weights[1]))(y)
            weight_index += 1 


        elif layer_list[i][0] == "activation": 
            if layer_list[i][1] == "square": 
                y = Activation(square_activation)(y) 
            elif layer_list[i][1] == "poly":
                y = Activation(poly_activation)(y) 

    return y 


##shallow neural network with a variable number of layers 
def cryptonets_model_no_conv(input, layer_list):

    def square_activation(x):
        return x * x
    def poly_activation(x): 
        return .125 * (x + 2) ** 2 - .25 

    y = Flatten()(input)

    dense_num = 0 
    for i in range(len(layer_list)):
        if layer_list[i][0] == "dense":
            name = "dense_" + str(dense_num)
            dense_num += 1
            y = Dense(layer_list[i][1], use_bias=True, name=name)(y) 
        elif layer_list[i][0] == "activation":
            if layer_list[i][1] == "square":
                y = Activation(square_activation)(y)
            elif layer_list[i][1] == "poly":
                y = Activation(poly_activation)(y) 
    return y 


#creates a hyperparameter search over valid architrectures with poly depth <= max_levels 
def generate_architectures(max_levels, input_size, output_size, include_poly = False, include_conv = False, min_levels=3, bit_precision=24): 

    #all architectures consist of linear layers plus activation functions 
    structures = [] 
    for i in range(min_levels, max_levels +1, 2):
        structure = []  
        for j in range(i): 
            name = "dense_like" if i % 2 == 0 else "activation"
            structure.append(name) 
        structures.append(structure)


    dense_options = ["dense", "bottleneck_dense"] 
    activation_options = ["square"]
    if include_poly:
        activation_options.append("poly")


    hidden_nodes_nums = {} 
    hidden_nodes_nums["dense"] = {} 
    hidden_nodes_nums["bottleneck_dense"] = {} 
    hidden_nodes_nums["dense"][0] = [i for i in range(100, 1100, 100)]
    hidden_nodes_nums["bottleneck_dense"][0] = [(a, b) for a in [25, 50, 100] for b in [100, 200, 400, 800, 1000]]


    #generate architectures 
    architectures = [] 

    for structure in structures: 
        for activation_option in activation_options: 
            constructing_architectures = [[]] 
            for i in range(len(structure)): 
                if structure[i] == "activation":
                    for j in constructing_architectures:
                        j.append(("activation", activation_option))
                elif structure[i] == "dense_like":
                    if i == len(structure) -1: #last layer 
                        for j in constructing_architectures: 
                            j.append(("dense", output_size))

                    else: 
                        new_archs = [] 
                        for arch in constructing_architectures:
                            for dense_option in dense_options: 
                                if dense_option == "dense":
                                    for num in hidden_nodes_nums["dense"][i]: 
                                        x = arch + [("dense", num)]
                                        new_archs.append(x) 
                                elif dense_option == "bottleneck_dense":
                                    for num1, num2 in hidden_nodes_nums["bottleneck_dense"][i]:
                                        x = arch + [("dense", num1)]
                                        x.append(("dense", num2))
                                        new_archs.append(x)
                        constructing_architectures = new_archs 

            architectures.extend(constructing_architectures) 

    return architectures 



def loss(labels, logits):
    return keras.losses.categorical_crossentropy(
        labels, logits, from_logits=True)

def logit_accuracy(y_true, y_pred_logit):


    y_test_label = tf.argmax(y_true, 1)
    y_pred = tf.argmax(y_pred_logit, 1)
    correct_prediction = tf.equal(y_pred, y_test_label)
    test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return test_accuracy 



def main(FLAGS):
    (x_train, y_train, y_train_label, x_test, y_test, y_test_label) = mnist_util.load_mnist_logit_data()

    #generate valid architectures for a given security level and fixed layer level:
    #architectures = generate_architecture(4096) TODO
    max_levels = 4
    input_size = 784
    output_size = 10 

    architectures = generate_architectures(max_levels, input_size, output_size)

    #[[("dense", 200), ("activation", "square"), ("dense", 10)]]
    # [("dense", 100), ("dense", 800), ("activation", "square"), ("dense", 10)], 
    # [("dense", 800), ("dense", 800), ("activation", "square"), ("dense", 10)],
    # [("dense", 800), ("dense", 800), ("dense", 10)]
    # [("dense", 800), ("activation", "square"), ("dense", 10)],
    # [("dense", 800), ("dense", 10), ("activation", "square")]]
    print("Architectures to test:")
    for i in range(len(architectures)):
        print(f"### ARCHITECTURE {i} ### ")
        print(architectures[i])

    
    accuracies = [] 
    for layer_list in architectures: 
        x = Input(
            shape=(
                28,
                28,
                1,
            ), name="input")
        y = cryptonets_model_no_conv(x, layer_list)
        cryptonets_model = Model(inputs=x, outputs=y)
        print(cryptonets_model.summary())

        optimizer = SGD(learning_rate=0.008, momentum=0.9)
        cryptonets_model.compile(
            optimizer=optimizer, loss='mean_squared_error', metrics=[logit_accuracy])

        cryptonets_model.fit(
            x_train,
            y_train,
            epochs=FLAGS.epochs,
            batch_size=FLAGS.batch_size,
            validation_data=(x_test, y_test),
            verbose=1)

        test_loss, test_acc = cryptonets_model.evaluate(x_test, y_test_label, verbose=1) #should this be y-test? No, evaluating against y_Test_label 
        print("Test accuracy:", test_acc)
        accuracies.append(test_acc)

        #reset graph 
        tf.reset_default_graph()
        sess = tf.compat.v1.Session()

    best_model = np.argmax(accuracies) 
    layer_list = architectures[best_model]

    print("best model", best_model) 

    x = Input(
        shape=(
            28,
            28,
            1,
        ), name="input")
    y = cryptonets_model_no_conv(x, layer_list)
    cryptonets_model = Model(inputs=x, outputs=y)
    print(cryptonets_model.summary())

    optimizer = SGD(learning_rate=0.008, momentum=0.9)
    cryptonets_model.compile(
        optimizer=optimizer, loss='mean_squared_error', metrics=[logit_accuracy])

    cryptonets_model.fit(
        x_train,
        y_train,
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        validation_data=(x_test, y_test),
        verbose=1)

    test_loss, test_acc = cryptonets_model.evaluate(x_test, y_test_label, verbose=1) #should this be y-test? No, evaluating against y_Test_label 
    print("Test accuracy:", test_acc)
    accuracies.append(test_acc)

    # Squash weights and save model


    weights, compressed_layer_list = squash_layers_variable(cryptonets_model,
                             tf.compat.v1.keras.backend.get_session(), layer_list)

    tf.reset_default_graph()
    sess = tf.compat.v1.Session()

    x = Input(
        shape=(
            28,
            28,
            1,
        ), name="input")
    y = cryptonets_model_no_conv_squashed(x, weights, compressed_layer_list)
    sess.run(tf.compat.v1.global_variables_initializer())

    mnist_util.save_model(
        sess,
        ["output/BiasAdd"],
        "./models",
        "compressnets_small",
    )


if __name__ == "__main__":
    FLAGS, unparsed = mnist_util.train_argument_parser().parse_known_args()
    if unparsed:
        print("Unparsed flags: ", unparsed)
        exit(1)

    main(FLAGS)
