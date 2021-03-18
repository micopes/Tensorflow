import tensorflow as tf
import numpy as np

print(tf.__version__)

t1 = tf.Variable([1, 2, 3])
t2 = tf.constant([1, 2, 3])

print(t1)
print(t2)

print("====")
print(type(t1))
print(type(t2))

# %%

test_list = [1, 2, 3]
test_np = np.array([1, 2, 3])

t1 = tf.constant(test_list)
t2 = tf.constant(test_np)

print(t1)
print(t2)

print(type(t1))
print(type(t2))

# %%
test_list = [1, 2, 3]
test_np = np.array([1, 2, 3])

t1 = tf.Variable(test_list)
t2 = tf.Variable(test_np)

print(t1)
print(t2)

print(type(t1))
print(type(t2))

# %%

t1 = tf.constant(test_list)
t2 = tf.Variable(test_list)

# t3 = tf.constant(t2)
t4 = tf.Varaible(t1)

# %%
t1 = tf.convert_to_tensor(test_list)
t2 = tf.convert_to_tensor(test_np)

t3 = tf.Variable(test_list)
t4 = tf.convert_to_tensor(t3)

print(type(t3))
print(type(t4))

# %%
t1 = tf.constant(test_list)
t2 = tf.constant(test_list)

t3 = t1+t2
print(type(t3))

# convert + convert => EagarTensor(Immutable)
# convert + Variable => EagarTensor(Immutable)
# Variable + Variable => EagarTensor(Immutable)

# 셋 다 EagarTensor로 Immutable한 것이 된다.