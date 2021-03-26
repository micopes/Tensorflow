import tensorflow as tf

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Layers, Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import SparseCategoricalAccuracy

class CustomLayer1(Layer):
    def __init__(self):
        super(CustomLayer1, self).__init__()
        
    def call(self, x):
    
model = Sequential()
model.add(CustomLayer1)