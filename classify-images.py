import numpy as np
from keras import Sequential
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

train_datagen = ImageDataGenerator(rescale=1 / 255)
validation_datagen = ImageDataGenerator(rescale=1 / 255)

# Note: separate images in to subdirectories by classification in the /image-train dir
train_generator = train_datagen.flow_from_directory(
    './images-train/',
    target_size=(512, 512),
    batch_size=16,
    class_mode='categorical'
)

# Note: separate images in to subdirectories by classification in the /image-test dir
validation_generator = validation_datagen.flow_from_directory(
    './images-test/',
    target_size=(512, 512),
    batch_size=16,
    class_mode='categorical'
)

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')
])

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    steps_per_epoch=6,
    epochs=50,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=6
)

# Note: example of separate images in subdirectories by classification in the /image-unseen dir
imageList = [
    'images-unseen/clifford/dragons-86080289-60d3-432d-9c6c-0c76ab8e6bba.png',
    'images-unseen/number-line-wave/neon-8-06f0cecb-6b9b-4f83-8eb3-00bff71210c4.png',
    'images-unseen/painful-needles/mocha-5e311905-9e77-4563-bab5-62118433c2c0.png',
    'images-unseen/spin/orange-v2-4b2fe0da-6518-4e26-ad95-72ff19351e1a.png',
    'images-unseen/tubes/constructivist-cd895928-439b-402e-9911-c8d2ca4b081d.png'
]

labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())

print(labels)

for path in imageList:
    img = image.load_img(path, target_size=(512, 512))
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)
    print(classes[0])


def plot_training_and_validation_accuracy_per_epoch(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', "Training Accuracy")
    plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
    plt.title('Training and validation accuracy')
    plt.show()


def plot_training_and_validation_loss_per_epoch(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'r', "Training Loss")
    plt.plot(epochs, val_loss, 'b', "Validation Loss")
    plt.show()


def plot_loss_acc(history):
    '''Plots the training and validation loss and accuracy from a history object'''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


plot_training_and_validation_accuracy_per_epoch(history)
plot_training_and_validation_loss_per_epoch(history)
plot_loss_acc(history)
