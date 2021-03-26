import tensorflow as tf
import matplotlib.pyplot as plt

X_train = tf.random.normal(shape = (1000, ), dtype = tf.float32)
y_train = 3*X_train + 1 + 0.2*tf.random.normal(shape = (1000, ), dtype = tf.float32)

X_test = tf.random.normal(shape = (300, ), dtype = tf.float32)
y_test = 3*X_test + 1 + 0.2*tf.random.normal(shape = (300, ), dtype = tf.float32)

# %% Sequential Method로 단순 모델 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units = 1,
                          activation = "linear") # Fully Connected
    
    ])

# compile하면 loss와 object가 만들어짐.
model.compile(loss = "mean_squared_error",
              optimizer = "SGD")

model.fit(X_train, y_train, epochs = 50, verbose = 2)
model.evaluate(X_test, y_test, verbose = 2)
 
# %% 시각화
fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(X_train.numpy(),
           y_train.numpy())
ax.tick_params(labelsize = 20)
ax.grid()



