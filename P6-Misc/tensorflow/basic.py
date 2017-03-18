import tensorflow as tf
import numpy as np

# define var
x = tf.constant(1, dtype=tf.int32)
y = tf.constant(2, dtype=tf.int32)

# build the graph
sum = tf.add(x, y)

# init
init = tf.global_variables_initializer()

# start to calcaluate
with tf.Session() as sess:
    sess.run(init)
    sess.run(sum)
    print(sum.eval())

# finish
print("End.")
