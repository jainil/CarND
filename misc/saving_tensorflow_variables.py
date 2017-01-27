import tensorflow as tf

save_file = 'model.cpkt'

weights = tf.Variable(tf.truncated_normal([2,3]))
bias = tf.Variable(tf.truncated_normal([3]))

saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  saver.save(sess, save_file)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  print('Weights: {}'.format(sess.run(weights)))
  print('Bias: {}'.format(sess.run(bias)))
  
  saver.restore(sess, save_file)

  print('Weights: {}'.format(sess.run(weights)))
  print('Bias: {}'.format(sess.run(bias)))


