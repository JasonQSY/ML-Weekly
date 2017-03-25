import numpy as np
import tensorflow as tf
import datetime

# Constants
n = 10

A = np.random.rand(5e3, 5e3).astype('float32')
B = np.random.rand(5e3, 5e3).astype('float32')

def matpow(M, n):
    if n < 1:
        return M
    else:
        return tf.matmul(M, matpow(M, n - 1))

with tf.device('/gpu:0'):
    a = tf.constant(A)
    b = tf.constant(B)
    m_sum = tf.add(matpow(a, n), matpow(b, n))


t1_1 = datetime.datetime.now()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(m_sum)

t1_2 = datetime.datetime.now()

print("GPU computation time: " + str(t1_2 - t1_1))

with tf.device('/cpu:0'):
    a = tf.constant(A)
    b = tf.constant(B)
    m_sum = tf.add(matpow(a, n), matpow(b, n))

t2_1 = datetime.datetime.now()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(m_sum)

t2_2 = datetime.datetime.now()

print("CPU computation time: " + str(t2_2 - t2_1))
