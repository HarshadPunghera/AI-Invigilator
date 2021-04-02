#!/usr/bin/env python
# coding: utf-8

# # VGG16 Pre Trained Model



# VGG16 is a pretrained model which contains with accuracy of 92.7 % using ImageNet which
# is a dataset of over 14 million images belonging to 1000 classes

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from glob import glob
#import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Flatten
from keras.applications import vgg16

# training the base model
# VGG16 model is trained on RGB images of size (224, 224), which is a default input size of the network.
# 3 represents the number of color channels (R,G,B)
# include_top states whether base classifier to be used or not
# weights refers to the pretrained weights for which argument is to be set as 'imagenet'.
# if building the model from scracth then weights need to be set as None
# pooling can be of 2 types max or avg. Pooling does the job of reducing the number of parameters and computation power.
# In most cases value used is 'max'.
# def train_model():

vgg16_model = vgg16.VGG16(input_shape=[244, 244] + [3],
                          include_top=False,
                          weights='imagenet',
                          pooling='max')
vgg16_model.summary()

# as all the layers have already been trained and features are already extracted, no need to re train the layers
# decision to re train the layers totally depend upon number of parameters and records in the dataset
# if datasize is too small i.e. less than 1000 and images class is not part of VGG dataset then
# its better to re train last few layers to extract features specific to the dataset

for layer in vgg16_model.layers:
    layer.trainable = False

# to get the number of classes (folders) inside train dataset

folder = glob("C:\\Users\\Viraj\\Capstone\\Train Images\\*")
folder

# flatten is used to convert convolutional layer 4D output to 2D output which is accepted by Dense layer
# Dense layer is a classifier which does the job of reducing 512 classes to 2

x = Flatten()(vgg16_model.output)
prediction = Dense(len(folder), activation='sigmoid')(x)

# optimizing the model

model = Model(inputs=vgg16_model.input, outputs=prediction)
model.summary()

# compiling the model

model.compile(loss="binary_crossentropy",
              optimizer='rmsprop',
              metrics=['accuracy'])

# Setting train path and validate path

train_path = 'C:\\Users\\Viraj\\Capstone\\Train Images'
valid_path = 'C:\\Users\\Viraj\\Capstone\\Test Images'

# to fit the model images needs to be in same size for which imagegenerate needs to be applied

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=(224, 224),
                                                    batch_size=16,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical')

# fitting the model

final_model = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
)

model.save('./models/vgg/final_model.h5')


