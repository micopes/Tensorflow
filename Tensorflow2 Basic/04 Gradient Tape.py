import tensorflow as tf
import matplotlib.pyplot as plt

'''
tf.GradientTape()
'''

t1 = tf.constant([1, 2, 3], dtype = tf.float32)
t2 = tf.Variable([10, 20, 30], dtype = tf.float32)

with tf.GradientTape() as tape:
    t3 = t1 * t2

gradients = tape.gradient(t3, [t1, t2])
print(type(gradients))

# t1 = tf.constant(..)로 설정해주면
# input값인 constant는 gradient 를 계산할 필요 없으므로 None이 된다.
print("dt1: ", gradients[0]) # t1에 대한 gradient
print("dt2: ", gradients[1]) # t2에 대한 gradient

# %%
x_data = tf.random.normal(shape = (1000, ), dtype = tf.float32)
# y = 3x + 1
y_data = 3*x_data + 1

# print(x_data.dtype, y_data.dtype)

w = tf.Variable(-1.)
b = tf.Variable(-1.)

learning_rate = 0.01
EPOCHS = 10

w_trace, b_trace = [], []
for epoch in range(EPOCHS):
    for x, y in zip(x_data, y_data):
        with tf.GradientTape() as tape:
            prediction = w*x + b # model
            loss = (prediction - y)**2 # loss object
        
        gradients = tape.gradient(loss, [w, b])
        
        w_trace.append(w.numpy())
        b_trace.append(b.numpy())
        w = tf.Variable(w - learning_rate * gradients[0])
        b = tf.Variable(b - learning_rate * gradients[1])
    
# %%
fig, ax = plt.subplots(figsize = (20, 10))

ax.plot(w_trace, label = "weight")
ax.plot(b_trace, label = "bias")

ax.tick_params(labelsize = 20)
ax.legend(fontsize = 30)
ax.grid()