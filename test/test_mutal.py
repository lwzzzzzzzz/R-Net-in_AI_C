import tensorflow as tf

# a = tf.ones([1,3,4])
# b = tf.random_normal([1,4,3])
#
# c = tf.squeeze(tf.matmul(a,b))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     ss = sess.run(c)
#     print(type(ss))
import numpy as np

a = np.zeros([2,3])
b = np.argmax(a, axis=1)
print(b)