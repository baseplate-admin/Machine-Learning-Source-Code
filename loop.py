import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist
(training_images , training_labels),(test_images,test_labels)=mnist.load_data()

model = tf.keras.models.load_model("test_model_1")
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(128,activation=tf.nn.relu),tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=99)

model.save("test_model_1")
