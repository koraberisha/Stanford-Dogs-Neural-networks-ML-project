# -*- coding: utf-8 -*-


import keras,os
from keras.models import Sequential
from keras.layers import Dropout,Dense, Conv2D,  MaxPooling2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add,Activation
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib as mpl
import cv2
import time
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.initializers import glorot_normal

from tensorflow.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras import applications
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import matplotlib.image as mpimg

tf.test.gpu_device_name()

from google.colab import drive
drive.mount('/content/drive')

train_data_dir = "/content/drive/MyDrive/Images"
img_width, img_height = 224, 224
channels = 3
batch_size = 64
num_images= 50
image_arr_size= img_width * img_height * channels

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

from keras import backend as K
import tensorflow_addons as tfa
from sklearn.metrics import f1_score
import sklearn




def preprocess_data(train_data_dir):
  train_datagen = ImageDataGenerator(rescale= 1./255,
    shear_range= 0.2,
    zoom_range= 0.2,
    horizontal_flip= True,
    rotation_range= 20,
    width_shift_range= 0.2,
    height_shift_range= 0.2,
    validation_split=0.2,)

  valid_datagen = ImageDataGenerator(
      rescale= 1./255,
      validation_split=0.2,
  )
  train_generator = train_datagen.flow_from_directory(
      train_data_dir,
      target_size= (img_width, img_height),
      color_mode= 'rgb',
      batch_size= batch_size,
      class_mode= 'categorical',
      subset='training',
      shuffle= True,
      seed= 1337
  )

  valid_generator = valid_datagen.flow_from_directory(
      train_data_dir,
      target_size= (img_width, img_height),
      color_mode= 'rgb',
      batch_size= batch_size,
      class_mode= 'categorical',
      subset='validation',
      shuffle= True,
      seed= 1337)


  return train_generator, valid_generator




def checkpoints():
  checkpoint = ModelCheckpoint('baseline_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto',
               save_weights_only=False, period=1  )

  earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=1, mode='auto')

  csvlogger = CSVLogger(filename= "training_csv.log", separator = ",", append = False)

  reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto')

  callbacks = [checkpoint, earlystop, csvlogger,reduceLR]
  return callbacks

def inceptionV3():
  InceptionV3 = applications.InceptionResNetV2(include_top= False, input_shape= (img_width, img_height, channels),
                                         weights= 'imagenet')

  model = Sequential()

  for layer in InceptionV3.layers:
      layer.trainable= False

  model.add(InceptionV3)
  model.add(GlobalAveragePooling2D())
  model.add(Dropout(0.2))
  model.add(Dense(120, activation='softmax'))

  model.compile(optimizer= keras.optimizers.Adam(lr= 0.0001), loss= 'categorical_crossentropy', metrics= ['accuracy'])
  return model


train_generator, valid_generator= preprocess_data('/content/drive/MyDrive/Images')
def main(train_generator, valid_generator):
  train_data_dir= '/content/drive/MyDrive/Images'

  callbacks= checkpoints()
  num_classes = len(train_generator.class_indices)
  train_labels = train_generator.classes
  train_labels = to_categorical(train_labels, num_classes=num_classes)
  valid_labels = valid_generator.classes
  valid_labels = to_categorical(valid_labels, num_classes=num_classes)
  nb_train_samples = len(train_generator.filenames)
  nb_valid_samples = len(valid_generator.filenames)



  model= inceptionV3()


  history = model.fit(train_generator, epochs = 30,batch_size=128, steps_per_epoch = nb_train_samples//batch_size,
                      validation_data = valid_generator, validation_steps = nb_valid_samples//batch_size, verbose = 1,
                      callbacks = callbacks, shuffle = True)




  model1 = keras.models.load_model('baseline_model.h5')
  model2 = keras.models.load_model('inception.h5')
  model3 = keras.models.load_model('xception epoch23.h5')


  model1.compile(optimizer= keras.optimizers.Adam(lr= 0.0001), loss= 'categorical_crossentropy', metrics=['accuracy',tf.keras.metrics.Recall(name='Recall'),
                           tf.keras.metrics.Precision(name='Precision'),
                           f1_m])
  model2.compile(optimizer= keras.optimizers.Adam(lr= 0.0001), loss= 'categorical_crossentropy', metrics=['accuracy',tf.keras.metrics.Recall(name='Recall'),
                           tf.keras.metrics.Precision(name='Precision'),
                           f1_m])
  model3.compile(optimizer= keras.optimizers.Adam(lr= 0.0001), loss= 'categorical_crossentropy', metrics=['accuracy',tf.keras.metrics.Recall(name='Recall'),
                           tf.keras.metrics.Precision(name='Precision'),
                           f1_m])

  h1 = model1.fit(train_generator, epochs = 1, steps_per_epoch = nb_train_samples//batch_size,
                      validation_data = valid_generator, validation_steps = nb_valid_samples//batch_size, verbose = 1,
                      callbacks = callbacks, shuffle = True)
  h2 = model2.fit(train_generator, epochs = 1, steps_per_epoch = nb_train_samples//batch_size,
                      validation_data = valid_generator, validation_steps = nb_valid_samples//batch_size, verbose = 1,
                      callbacks = callbacks, shuffle = True)
  h3 = model3.fit(train_generator, epochs = 1, steps_per_epoch = nb_train_samples//batch_size,
                      validation_data = valid_generator, validation_steps = nb_valid_samples//batch_size, verbose = 1,
                      callbacks = callbacks, shuffle = True)



  y_pred1 = model1.predict(valid_generator)
  y_pred2 = model2.predict(valid_generator)
  y_pred3 = model3.predict(valid_generator)
  predicted_categories1 = np.argmax(y_pred1, axis = 1)
  predicted_categories2 = np.argmax(y_pred2, axis = 1)
  predicted_categories3 = np.argmax(y_pred3, axis = 1)

  true_categories = tf.concat([y for x, y in valid_generator], axis = 0).numpy()
  true_categories_argmax = np.argmax(true_categories, axis = 1)

  print(sklearn.metrics.classification_report(true_categories_argmax, predicted_categories1))
  print(sklearn.metrics.classification_report(true_categories_argmax, predicted_categories2))
  print(sklearn.metrics.classification_report(true_categories_argmax, predicted_categories3))



main(train_generator, valid_generator)
loss1, accuracy1, precision1, recall1, f1_score1 = model1.evaluate(valid_generator, verbose=1)
loss2, accuracy2, precision2, recall2, f1_score2 = model2.evaluate(valid_generator, verbose=1)
loss3, accuracy3, precision3, recall3, f1_score3 = model3.evaluate(valid_generator, verbose=1)

print("InceptionResnetV2")
print("Loss: " + str(loss1) + "  Accuracy: " + str(accuracy1) + "  F1: " + str(f1_score1) + "  Precision: " + str(precision1) + "  Recal: " + str(recall1))
print()
print("InceptionV3")
print("Loss: " + str(loss2) + "  Accuracy: " + str(accuracy2) + "  F1: " + str(f1_score2) + "  Precision: " + str(precision2) + "  Recal: " + str(recall2))
print()
print("Xception")
print("Loss: " + str(loss3) + "  Accuracy: " + str(accuracy3) + "  F1: " + str(f1_score3) + "  Precision: " + str(precision3) + "  Recal: " + str(recall3))
