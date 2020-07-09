import os

ask_base_dir=input("What is your base extracted zip file directory?")
train_dir=os.path.join(ask_base_dir,"train")
validation_dir = os.path.join(ask_base_dir,"validation")

ask_directory_folder=input("How many training variable are there?")

import tensorflow as tf
import tensorflow.keras

def model(number):
    print("This is Convolutional Neural Network")
    print("""
    Which Neural network will you choose?
    """)
    if number==1:
        print("""
        This is number 1 model.
        Contains These Settings.
        tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
        """)