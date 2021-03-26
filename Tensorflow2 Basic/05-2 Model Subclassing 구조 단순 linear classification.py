import tensorflow as tf
import matplotlib.pyplot as plt

X_train = tf.random.normal(shape = (10, 1), dtype = tf.float32)
y_train = 3*X_train + 1 + 0.2*tf.random.normal(shape = (10, 1), dtype = tf.float32)

X_test = tf.random.normal(shape = (3, 1), dtype = tf.float32)
y_test = 3*X_test + 1 + 0.2*tf.random.normal(shape = (3, 1), dtype = tf.float32)

# %% Model Subclassing 으로 모델 구성

from termcolor import colored
class LinearPredictor(tf.keras.Model):
    def __init__(self):
        super(LinearPredictor, self).__init__()
        
        self.d1 = tf.keras.layers.Dense(units = 1,
                                        activation = "linear")
    def call(self, x):
        x = self.d1(x)
        return x

EPOCHS = 10
LR = 0.01

# instantiation learning objects
model = LinearPredictor()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate = LR)

for epoch in range(EPOCHS):
    for x, y in zip(X_train, y_train):
        x = tf.reshape(x, (1, 1))
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
        
        # 500개에 대해서 하면 500개 각각의 w, b. gradient 이용
        gradients = tape.gradient(loss, model.trainable_variables)
        # update
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      
    print(colored("Epoch: ", "red", "on_white"), epoch + 1)
    
    template = "Train Loss {}"
    print(template.format(loss))
 
# %% 시각화
fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(X_train.numpy(),
           y_train.numpy())
ax.tick_params(labelsize = 20)
ax.grid()



