import os

base_dir='tmp/'
train_dir= os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')

cats_dir= os.path.join(train_dir,'cats')
dogs_dir= os.path.join(train_dir,'dogs')

cats_dir_val=os.path.join(validation_dir,'cats')
dogs_dir_val=os.path.join(validation_dir,'dogs')

fname_cats=os.listdir(cats_dir)
fname_dogs=os.listdir(dogs_dir)

print('Total cats image:',len(fname_cats))
print('Total dogs Image:',len(fname_dogs))

import tensorflow as tf
import tensorflow.keras


model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.summary()


model.compile(optimizer = 'adam',
loss="binary_crossentropy",metrics = ['accuracy'])

import tensorflow.keras.preprocessing.image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen= ImageDataGenerator(rescale=1.0/255.0)
validation_datagen= ImageDataGenerator(rescale=1.0/255.0)

train_generator=train_datagen.flow_from_directory(
train_dir,
target_size=(150,150),
class_mode='binary',
batch_size=64
)
validation_generator=validation_datagen.flow_from_directory(
validation_dir,
target_size=(150,150),
class_mode='binary',
batch_size=64
)
history=model.fit(train_generator,
        validation_data=validation_generator,
        steps_per_epoch=100,
        epochs=25,
        validation_steps=50,
        verbose=0)
