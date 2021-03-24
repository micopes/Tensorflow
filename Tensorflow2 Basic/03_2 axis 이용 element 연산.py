import tensorflow as tf
import numpy as np

t1 = tf.random.normal(shape = (3, 4), mean = 0, stddev = 5)
t1 = tf.cast(t1, dtype = tf.int16)

# %% 
'''
tf.reduce_sum
tf.reduce_prod

tf.reduce_max
tf.reduce_min

tf.reduce_mean
tf.reduce_std
tf.reduce_variance

tf.reduce_all
tf.reduce_any
'''

# %%
t1 = tf.random.normal(shape = (3, 4), mean = 0, stddev = 5)
t1 = tf.cast(t1, dtype = tf.int16)

t2 = tf.reduce_sum(t1)
print(t1.numpy())
print(t2.numpy())

# %%
t2 = tf.reduce_sum(t1, axis = 0)

print(t1.numpy())
print(t2.numpy())
print(t2)

# %%
t2 = tf.reduce_sum(t1, axis = 1)
print(t1.numpy())
print(t2.numpy())
print(t1)

# %%
t1 = tf.random.normal(shape = (128, 128, 3), mean = 0, stddev = 5)
t1 = tf.cast(t1, dtype = tf.int16)

t2 = tf.reduce_sum(t1, axis = 2)
print(t1.shape)
print(t2.shape)