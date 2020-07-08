import os
from zipfile import ZipFile

ask_if_zip_extract=input("""
Is your zip already extracted?
If its not extracted enter "No" or "no"
Otherwise press enter.
""")

if ask_if_zip_extract == "No" or ask_if_zip_extract =="no":
    training_zip = 'rps.zip'
    with ZipFile(training_zip, 'r') as zip:
        zip.printdir()
        zip.extractall()
    test_zip='rps-test-set.zip'
    with ZipFile(test_zip, 'r') as zip:
        zip.printdir()
        zip.extractall()

rock_dir=('rps/rock')
paper_dir=('rps/paper')
scissor_dir=('rps/scissors')

print("Total Rock images: ", len(os.listdir(rock_dir)))
print("Total Paper images: ", len(os.listdir(paper_dir)))
print("Total Scissors images: ", len(os.listdir(scissor_dir)))

rock_files= os.listdir(rock_dir)
print(rock_files[:10])

paper_files=os.listdir(paper_dir)
print(paper_files[:10])

scissor_files=os.listdir(scissor_dir)
print(scissor_files[:10])



import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR="rps/"
training_datagen =  ImageDataGenerator(
        rescale = 1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
)

VALIDATION_DIR="rps-test-set/"
validation_datagen= ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=126
)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=126
)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


history=model.fit(train_generator,
                validation_data=validation_generator,
                setps_per_epoch=10,
                validation_steps=3,
                verbose=1
                )
model.save("rpc.h5")
