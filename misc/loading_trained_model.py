import tensorflow as tf
import numpy as np
import math
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
n_input = 784
n_classes = 10

mnist = input_data.read_data_sets('.', one_hot=True)

features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

logits = tf.add(tf.matmul(features, weights), bias)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


save_file = 'train_model.ckpt'

saver = tf.train.Saver()

with tf.Session() as sess:
  saver.restore(sess, save_file)

  test_accuracy = sess.run(accuracy, feed_dict={
          features: mnist.validation.images,
          labels: mnist.validation.labels
        })

  print('Test Accuracy: {}'.format(test_accuracy))
