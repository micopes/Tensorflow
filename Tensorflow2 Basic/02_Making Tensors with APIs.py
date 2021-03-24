import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

t2 = tf.ones(shape = (100, 3))
print(t2)

t3 = tf.zeros(shape = (128, 128, 3))
print(t3)

# %%

PI = np.pi
t4 = 0 * tf.ones(shape = (128, 128, 3))
print(t4)

# %%

test_list = [[1, 2, 3], [4, 5, 6]]

t1 = tf.Variable(test_list)
print(t1)

# t2 = tf.ones(shape = t1.shape)
t2 = 3*tf.ones_like(t1)
print(t2)

# %%

np.random.seed(0)
tf.random.set_seed(0) # np.random.seed(0)과 동일

t1 = tf.random.normal(shape = (3, 3))
print(t1)

# %%
# 평균 3이고, 표준편차 1인, 랜덤하게 생성.
t2 = tf.random.normal(mean = 3, stddev = 1, shape = (1000, ))
print(t2)

fig, ax = plt.subplots(figsize = (15, 15))
ax.hist(t2.numpy(), bins = 30)

ax.tick_params(labelsize = 20)

# %%

t2 = tf.random.uniform(shape = (10000, ), min val = -10, maxval = 10)
print(t2)

fig, ax = plt.subplots(figsize = (15, 15))
ax.hist(t2.numpy(), bins = 30)

ax.tick_params(labelsize = 20) 

# %%
t2 = tf.random.poisson(shape = (1000, ), lam = 5) 
print(t2)

fig, ax = plt.subplots(figsize = (15, 15))
ax.hist(t2.numpy(), bins = 30)

ax.tick_params(labelsize = 20) 

# %%
# 기본 타입은 'tf.float32'
t1 = tf.random.normal(shape = (128, 128, 3))
print("t1.shape:", t1.shape)
print("t1.dtype:", t1.dtype)

# %%
test_np = np.random.randint(, 1, size = (100, ))
print(test_np.dtype)

# t1 = tf.constant(test_np)
# 'tf.int32' 타입
# 텐서 연산을 할 때 타입이 안맞으면 문제가 발생할 수 있으므로 
# 아래와 같이 dtype = float32로 설정해준다.
t1 = tf.constant(test_np, dtype = tf.float32)
print(t1.dtype)