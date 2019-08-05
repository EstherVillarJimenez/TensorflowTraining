# TensorflowTraining

import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))

# TensorflowWeek1HouseModel

import tensorflow as tf
import numpy as np
from tensorflow import keras

def house_model(y_new):
    
    xs = np.array([1.0,2.0,3.0,4.0,5.0,6.0], dtype=float)
    ys = np.array([1.0,1.5,2.0,2.5,3.0,3.5], dtype=float)
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=500)
    return model.predict(y_new)[0]
    
prediction = house_model([7.0])
print(prediction)
