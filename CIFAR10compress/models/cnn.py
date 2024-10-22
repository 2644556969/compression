from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model

WEIGHT_DECAY = 1e2

'''
3 different model architectures for shallow CNN in model compression
'''



class CNN(model.Model):
    def __init__(self,
                 train_poly_act,
                 batch_norm,
                 wd=WEIGHT_DECAY,
                 training=True):
        super(CNN, self).__init__(
            model_name='cnn',
            wd=wd,
            training=training,
            train_poly_act=train_poly_act,
            batch_norm=batch_norm)

    def inference(self, images):
        conv1 = self.conv_layer(
            images,
            size=5,
            filters=40,
            stride=2,
            decay=True,
            activation=True,
            batch_norm=self.batch_norm,
            name='conv1')

        pool1 = self.pool_layer(conv1, size=5, stride=2, name='pool1')

        conv2 = self.conv_layer(
            pool1,
            size=3,
            filters=80,
            stride=1,
            decay=True,
            activation=True,
            batch_norm=self.batch_norm,
            name='conv2')

        fc1 = self.fc_layer(
            conv2,
            neurons=10,
            decay=True,
            activation=False,
            batch_norm=False,
            name='fc1')

        return fc1


class smallCNN(model.Model):
    def __init__(self,
                 train_poly_act,
                 batch_norm,
                 wd=WEIGHT_DECAY,
                 training=True):
        super(smallCNN, self).__init__(
            model_name='small_cnn',
            wd=wd,
            training=training,
            train_poly_act=train_poly_act,
            batch_norm=batch_norm)

    def inference(self, images):
        #SMALL CNN ARCHITECTURE: 
        #conv, bn, act, (3), avg pool (4), dense (5) 
        conv1 = self.conv_layer(
            images,
            size=5,
            filters=40,
            stride=2,
            decay=True,
            activation=True,
            batch_norm=self.batch_norm,
            name='conv1')

        #performs average pooling 
        pool1 = self.pool_layer(conv1, size=5, stride=2, name='pool1')

        # conv2 = self.conv_layer(
        #     pool1,
        #     size=3,
        #     filters=80,
        #     stride=1,
        #     decay=True,
        #     activation=True,
        #     batch_norm=self.batch_norm,
        #     name='conv2')

        fc1 = self.fc_layer(
            pool1,
            neurons=10,
            decay=True,
            activation=False,
            batch_norm=False,
            name='fc1')

        return fc1

#this differs from smallCNN in that it has an activation layer afterwards (is it possibl eto do square? to save a layer)
class smallCNNAct(model.Model):
    def __init__(self,
                 train_poly_act,
                 batch_norm,
                 wd=WEIGHT_DECAY,
                 training=True):
        super(smallCNNAct, self).__init__(
            model_name='small_cnnact',
            wd=wd,
            training=training,
            train_poly_act=train_poly_act,
            batch_norm=batch_norm)

    def inference(self, images):
        #SMALL CNN ARCHITECTURE: 
        #conv, bn, act, (3), avg pool (4), dense (5), act (7)  
        conv1 = self.conv_layer(
            images,
            size=5,
            filters=40,
            stride=2,
            decay=True,
            activation=True,
            batch_norm=self.batch_norm,
            name='conv1')

        #performs average pooling 
        pool1 = self.pool_layer(conv1, size=5, stride=2, name='pool1')

        # conv2 = self.conv_layer(
        #     pool1,
        #     size=3,
        #     filters=80,
        #     stride=1,
        #     decay=True,
        #     activation=True,
        #     batch_norm=self.batch_norm,
        #     name='conv2')

        fc1 = self.fc_layer(
            pool1,
            neurons=10,
            decay=True,
            activation=True,
            batch_norm=False,
            name='fc1')

        return fc1