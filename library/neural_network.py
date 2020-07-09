

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.preprocessing.image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def model(number,train_dir,validation_dir):
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
        input_shape_ask_height=input("What is your image height?")
        input_shape_ask_width=input("What is your Image Width?")
        input_ask_binary_or_categorical=input("""
        Is your image binary or categorical?
        If binary Press 1
        If Categorical Press 2.
        """)
        if input_ask_binary_or_categorical==1:
            activation_dense='sigmoid'
            crossentropy='binary_crossentropy'
            class_1='binary'
        elif input_ask_binary_or_categorical==2:
            activation_dense='softmax'
            crossentropy='categorical_crossentropy'
            class_1='categorical'
        def models():
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(input_shape_ask_width,input_shape_ask_height,3)),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512,activation='relu'),
                tf.keras.layers.Dense(1, activation=activation_dense)
            ])
            model.summary()
            model.compile(optimizer = 'adam',
            loss=crossentropy,metrics = ['accuracy'])
            
            
            train_datagen= ImageDataGenerator(rescale=1.0/255.0)
            validation_datagen= ImageDataGenerator(rescale=1.0/255.0)
            train_generator=train_datagen.flow_from_directory(
                train_dir,
                target_size=(input_shape_ask_width,input_shape_ask_height),
                class_mode=class_1,
                batch_size=64
            )
            validation_generator=validation_datagen.flow_from_directory(
                validation_dir,
                target_size=(input_shape_ask_width,input_shape_ask_height),
                class_mode=class_1,
                batch_size=64
            )
            history=model.fit(train_generator,
            validation_data=validation_generator,
            steps_per_epoch=100,
            epochs=25,
            validation_steps=50,
            verbose=0)