######################################################################3
import os
import cv2
from keras.preprocessing.image import img_to_array
#import glob
train_data = []
train_labels_one_hot =[]
trainfileList =[]

for root, dirs, files in os.walk("C:\\SAProject\\Vid\\train"):
    for file in files:
        if file.endswith(".jpg"):
             fullName = os.path.join(root, file)
             trainfileList.append(fullName)


for imagePath in trainfileList:
    
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image)
    train_data.append(image)
    path = os.path.split(imagePath)
    x = path[0]
    label = x.split("\\")[4]
    train_labels_one_hot.append(label)
    print(imagePath)
    print(label)
    
  
#################################TEST data######################################################3

import os
import cv2
from keras.preprocessing.image import img_to_array
#import glob
test_data = []
test_labels_one_hot =[]
testfileList =[]

for root, dirs, files in os.walk("C:\\SAProject\\Vid\\test"):
    for file in files:
        if file.endswith(".jpg"):
             fullName = os.path.join(root, file)
             testfileList.append(fullName)


for imagePath in testfileList:
    
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image)
    test_data.append(image)
    path = os.path.split(imagePath)
    x = path[0]
    label = x.split("\\")[4]
    test_labels_one_hot.append(label)
    print(imagePath)
    print(label)
    

########################################################################################


#from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
import pandas as pd

batch_size = 32
num_classes = 12
epochs = 100
data_augmentation = True
num_predictions = 20
#save_dir = os.path.join(os.getcwd(), 'saved_models')
#model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
train_data = np.array(train_data)
test_data = np.array(test_data)
print('train shape:', train_data.shape)
print(train_data.shape[0], 'train samples')
print(test_data.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
train_labels_unique= pd.Series(train_labels_one_hot).drop_duplicates().tolist()
test_labels_unique = pd.Series(test_labels_one_hot).drop_duplicates().tolist()

train_labels_one_hot = np.array(train_labels_one_hot)
test_labels_one_hot = np.array(test_labels_one_hot)
#print(test_labels_one_hot.shape)

#train_label = train_labels_one_hot
#test_label = test_labels_one_hot
#import tensorflow as tf
#train_label = tf.keras.utils.to_categorical(tf.keras.preprocessing.text.one_hot(np.array(train_labels_unique),12))
#test_label = keras.utils.to_categorical(test_labels_unique, num_classes=12)

print(train_data.shape)
print(test_data.shape)
print(train_labels_one_hot.shape)
print(test_labels_one_hot.shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=train_data.shape[1:]))
#model.add(Flatten(input_shape=train_data.shape[1:]))
 
 
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(train_data, train_labels_one_hot,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(test_data, test_labels_one_hot),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(train_data)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(train_data, train_label,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(test_data, test_label),
                        workers=4)

# Save model and weights
#if not os.path.isdir(save_dir):
#    os.makedirs(save_dir)
#model_path = os.path.join(save_dir, model_name)
#model.save(model_path)
#print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(test_data, test_label, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#model.save("CBA-model.h5")