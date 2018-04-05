import numpy as np
import tensorflow as tf
from math import sqrt

def conv_layer(input, filter_sz, stride=1, dropout=1.0):
    # convolution
    stddev = 1 / sqrt(filter_sz[0] * filter_sz[1] * filter_sz[2])
    out = tf.nn.conv2d(
        input=input,
        filter=tf.Variable(tf.truncated_normal(filter_sz, stddev=stddev)),
        strides=[1, stride, stride, 1],
        padding='SAME',
    )
    out = tf.nn.bias_add(out, tf.zeros([filter_sz[-1]]))

    # batch norm
    shape = out.get_shape().as_list()
    out_channels = shape[-1]
    mean, var = tf.nn.moments(out, [0])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = tf.Variable(tf.truncated_normal([out_channels], stddev=0.1))
    out = tf.nn.batch_norm_with_global_normalization(
            out, mean, var, beta, gamma, 0.001,
            scale_after_normalization=True)

    # activation
    out = tf.nn.relu(out)

    # dropout
    out = tf.nn.dropout(out, dropout)
    return out

def max_pool(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def dense_layer(X, in_dim, out_dim, var, relu=True, dropout=1.0):
    weight = tf.Variable(tf.random_normal([in_dim, out_dim], stddev=sqrt(var)))
    bias = tf.zeros([out_dim])
    output = tf.add(tf.matmul(X, weight), bias)
    if relu:
        output = tf.nn.relu(output)
    output = tf.nn.dropout(output, dropout)
    return output

def model(X):
    X = tf.random_crop(X, [128, 32, 32, 3])
    # (32x32x3)
    X = conv_layer(X, filter_sz=[3, 3, 3, 64], dropout=0.8)
    X = conv_layer(X, filter_sz=[3, 3, 64, 64], dropout=0.8)
    X = max_pool(X)
    # (16x16x64)
    X = conv_layer(X, filter_sz=[3, 3, 64, 128], dropout=0.8)
    X = conv_layer(X, filter_sz=[3, 3, 128, 128], dropout=0.8)
    X = max_pool(X)
    # (8x8x128)
    X = conv_layer(X, filter_sz=[3, 3, 128, 256])
    X = conv_layer(X, filter_sz=[3, 3, 256, 256])
    X = conv_layer(X, filter_sz=[3, 3, 256, 256])
    X = max_pool(X)
    # (4x4x256)
    X = conv_layer(X, filter_sz=[3, 3, 256, 512])
    X = conv_layer(X, filter_sz=[3, 3, 512, 512])
    X = conv_layer(X, filter_sz=[3, 3, 512, 512])
    X = max_pool(X)
    # (2x2x512)
    X = conv_layer(X, filter_sz=[3, 3, 512, 512])
    X = conv_layer(X, filter_sz=[3, 3, 512, 512])
    X = conv_layer(X, filter_sz=[3, 3, 512, 512])
    X = max_pool(X)
    # (1x1x512)
    X = tf.reshape(X, [-1, 1 * 1 * 512])
    X = dense_layer(X, in_dim=512, out_dim=100, var=1/512)
    X = dense_layer(X, in_dim=100, out_dim=4, var=1/512, relu=False)
    return X
