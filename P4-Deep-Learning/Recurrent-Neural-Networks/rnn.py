import tensorflow as tf
import numpy as np

# read data
data = open('input.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# constants
hidden_size = 100
seq_length = 25
sample_length=200
learning_rate = 1e-2
forget_bias = 0.5 # just for LSTM

# define input and output
x = tf.placeholder(dtype=tf.float32, shape=[None, vocab_size])
y = tf.placeholder(dtype=tf.float32, shape=[None, vocab_size])

# define weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([hidden_size, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

# build graph
def RNN(inputs, weights, biases):
    inputs = tf.split(inputs, 1)
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=forget_bias)
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, inputs, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

# transform [2] => [0, 1, 0, ...]
def one_hot_trans(list, length=seq_length, size=vocab_size):
    matrix = np.zeros((length, size))
    for i in range(len(list)):
        matrix[i][list[i]] = 1

    return np.array(matrix)

# produce sample text based on the network
def produce_text(initial, sess, pred, length=sample_length):
    pred_a = 0
    output = [initial]
    for i in range(length):
        if i == 0:
            a = one_hot_trans([initial], 1)
        else:
            a = one_hot_trans([pred_a], 1)

        p = sess.run(prob, feed_dict={x: a})
        p = p / np.sum(p) # make np.sum(p) = 1
        decision = np.random.choice(range(vocab_size), p=p)
        output.append(decision)

    txt = ''.join(ix_to_char[ix] for ix in output)
    print(txt)

# define ops
pred = RNN(x, weights, biases) # predict show value for each character
prob = tf.nn.softmax(pred[0]) # add sigmoid function -> (0, 1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()

# run
with tf.Session() as sess:
    sess.run(init)

    # training
    for i in range(50):
        p = 0
        while p + seq_length < data_size:
            batch_x = one_hot_trans([char_to_ix[ch] for ch in data[p : p + seq_length]])
            batch_y = one_hot_trans([char_to_ix[ch] for ch in data[p + 1 : p + 1 + seq_length]])
            optimizer.run(feed_dict={x: batch_x, y: batch_y})
            p += seq_length

        if i % 5 == 0:
            batch_x = one_hot_trans([char_to_ix[ch] for ch in data[0 : data_size - 2]], length=data_size - 1)
            batch_y = one_hot_trans([char_to_ix[ch] for ch in data[1 : data_size - 1]], length=data_size - 1)
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Loss function: %f" % (loss))
            initial = np.random.randint(vocab_size)
            produce_text(initial, sess, pred)

