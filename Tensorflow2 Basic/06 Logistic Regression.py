import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

plt.style.use('seaborn')
n_sample = 100
# .astype 이용해서 타입 변경
X_train = np.random.normal(0, 1, size = (n_sample, 1)).astype(np.float32)
y_train = (X_train >= 0).astype(np.float32)

print(y_train)

# %%
class classifier(tf.keras.Model):
    def __init__(self):
        super(classifier, self).__init__()
        
        self.d1 = tf.keras.layers.Dense(units = 1,
                                        activation = 'sigmoid')
    
    def call(self, x):
        prediction = self.d1(x)
        return prediction
    
EPOCHS = 10
LR = 0.01
model = classifier()
loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate = LR)

loss_metric = tf.keras.metrics.Mean() # 데이터셋 전체의 평균.
acc_metric = tf.keras.metrics.CategoricalAccuracy()

for epoch in range(EPOCHS):
    for x, y in zip(X_train, y_train):
        x = tf.reshape(x, (1, 1))
        
        # forward propagation
        with tf.GradientTape() as tape:
            prediction = model(x)
            loss = loss_object(y, prediction)
        
        # gradient 
        gradients = tape.gradient(loss, model.trainable_variables)
        # parameter
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # dataset 전체의 평균
        loss_metric(loss)
        # 맞는지 아닌지를 구하는 것, y와 y_pred가 얼마나 맞은지를 확인해줌.
        acc_metric(y, prediction)
        
    print(colored("Epoch: ", 'cyan', 'on_white'), epoch + 1)
    template = "Train loss {:.4f}\t Train Accuracy: {:.2f}%"
    
    ds_loss = loss_metric.result() # 전체 결과의 평균
    ds_acc = acc_metric.result() # 전체 결과의 정확도
    
    print(template.format(ds_loss, 
                          ds_acc*100))
    
    # loss_metric.reset.states()
    # acc_metric.reset.states()
                                  
# %%
X_min, X_max = X_train[0], X_train[1]

X_test = np.linspace(X_min, X_max, 300).astype(np.float32).reshape(-1, 1)

X_test_tf = tf.constant(X_test)
y_test_tf = model(X_test_tf)

x_result = X_test_tf.numpy()
y_result = y_test_tf.numpy()


# %% 시각화
fig, ax = plt.subplots(figsize = (20, 10))
ax.scatter(X_train, y_train)
ax.tick_params(labelsize = 20)
ax.plot(x_result, y_result,
        'r:',
        linewidth = 3)

                         

