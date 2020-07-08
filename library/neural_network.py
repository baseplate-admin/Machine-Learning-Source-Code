import os

ask_base_dir=input("What is your base extracted zip file directory?")
train_dir=os.path.join(ask_base_dir,"train")
validation_dir = os.path.join(ask_base_dir,"validation")

ask_directory_folder=input("How many training variable are there?")

import tensorflow as tf
import tensorflow.keras

def model(number):
    print("This is Convolutional Neural Network")
    ask_for_neural_input=int(input("""
    Which Neural network will you choose?
    """)
