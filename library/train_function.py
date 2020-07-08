def cat_and_dogs():
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
    model.save("cad.h5")
def rps():
    import os

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
            steps_per_epoch=100,
            epochs=25,
            validation_steps=50,
            verbose=0)
    model.save("rps.h5")

def tensorflow_mnist():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import matplotlib.pyplot as plt

    mnist = tf.keras.datasets.fashion_mnist
    (training_images , training_labels),(test_images,test_labels)=mnist.load_data()
    training_images=training_images/255
    test_images=test_images/255

    plt.imshow(training_images[0])
    print(training_labels[0])
    print(training_images[0])

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128,activation=tf.nn.relu),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(training_images, training_labels, epochs=5)

    model.save("test_model_1.h5")
