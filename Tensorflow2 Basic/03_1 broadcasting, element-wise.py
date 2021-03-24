import tensorflow as tf
import numpy as np

# Element-wise
t1 = tf.constant([1, 2, 3])
t2 = tf.constant([10, 20, 30])

print(t1 + t2)
print(t1 - t2)
print(t1 * t2)
print(t1 / t2)
print(t1 % t2)
print(t1 // t2)

# Broadcasting
# %%
t1 = tf.random.normal(shape = (3, 4), mean = 0, stddev = 5)
t2 = tf.random.normal(shape = (3, 1), mean = 0, stddev = 5)

t1 = tf.cast(t1, dtype = tf.int16)
t2 = tf.cast(t2, dtype = tf.int16)

t3 = t1 + t2

print(t1.numpy())
print(t2.numpy())
print(t3.numpy())

# %%
t1 = tf.random.normal(shape = (3, 4), mean = 0, stddev = 5)
t2 = tf.random.normal(shape = (1, 4), mean = 0, stddev = 5)

t1 = tf.cast(t1, dtype = tf.int16)
t2 = tf.cast(t2, dtype = tf.int16)

t3 = t1 + t2

print(t1.numpy())
print(t2.numpy())
print(t3.numpy())

# %%
t1 = tf.random.normal(shape = (1, 4), mean = 0, stddev = 5)
t2 = tf.random.normal(shape = (3, 1), mean = 0, stddev = 5)

t1 = tf.cast(t1, dtype = tf.int16)
t2 = tf.cast(t2, dtype = tf.int16)

t3 = t1 + t2

print(t1.numpy())
print(t2.numpy())
print(t3.numpy())