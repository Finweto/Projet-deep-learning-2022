# -*- coding: utf-8 -*-
"""
CNN_bibli.py
# 02/12/2022
"""


############################################################## IMPORTS ##############################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import random
import zipfile
import shutil
from shutil import copyfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib

largeur_image = ???
longueur_image = ???
nb_channel = 1


############################################################## PARSING ##############################################################
def parse_data_from_input(filename, training_size, ligne_training_csv, nb_pixels_images, n_classes):
    
  with open(filename) as file:
    labels = []
    img = []
  
    CSVData = open(filename)
    array = np.loadtxt(CSVData, delimiter=",")
    
    # mise de tous les labels dans un tableau 2D label[[labels images traing 81][labels images traing 82] ...]
    for bigLine in (training_size):
      for line in range(nb_pixels_images, ligne_training_csv, (nb_pixels_images+6)):
          labels.append(array[line][bigLine])
    labels = np.array(labels).reshape(training_size*n_classes, 6)
      
    # mise des images de training dans un tableau 3D 
    for i in range(n_classes):
        for bigLine in range(training_size):
          for line in range(i*(nb_pixels_images+6), (i*(nb_pixels_images+2))+nb_pixels_images, 1):
            img.append(array[line][bigLine])
        img2 = np.expand_dims(img, axis=0)
    images = img2.reshape(training_size*n_classes, largeur_image, longueur_image)

    return images, labels


############################################################## DATA AUGMENTATION ##############################################################
def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 25
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 30.196376257448144)
    return img

# ---------------------------------------------------------------------------------------------------------------------------------------------

def simple_train_val_generators(training_images, training_labels, training_size, n_classes):
  training_images = training_images.reshape(training_size*n_classes, largeur_image, longueur_image, nb_channel) 
 21, 1)
  
  train_datagen = ImageDataGenerator(
      rescale = 1./255.,
      shear_range=0.2,
      zoom_range=0.2,
      preprocessing_function=add_noise,
      horizontal_flip=True)
  train_generator = train_datagen.flow(x=training_images,
                                       y=training_labels,
                                       batch_size=50) 

  return train_generator


############################################################## ARCHITECTURES CNNs ##############################################################
class myCallback(tf.keras.callbacks.Callback): 
  # early stopping 
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy')>0.976):
      self.model.stop_training = True

# ---------------------------------------------------------------------------------------------------------------------------------------------

def simple_train_val_generators(n_classes):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(512, (3,3), activation='relu', input_shape = (largeur_image, longueur_image, nb_channel)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(n_classes+1, activation='softmax')
  ])
  model.summary()
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

# ---------------------------------------------------------------------------------------------------------------------------------------------

def run_model(train_generator, validation_generator, n_classes):
  callbacks = myCallback()
  model = create_model(n_classes)
  
  def on_epoch_end(self, epoch, logs={}):
    print(logs.get('val_accuracy'))

  history = model.fit(train_generator,
                      epochs=50, 
                      verbose=1,
                      callbacks=callbacks)
  return model


############################################################## PREDICTIONS ET EVALUATIONS #####################################################

def csvWriter(fil_name, nparray):
  example = nparray.tolist()
  with open(fil_name+'.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(example)

# ---------------------------------------------------------------------------------------------------------------------------------------------

def prediction_1_image(filename, indice_li, indice_co, nb_pixels_images, model, n_classes):
  CSVData = open(filename)
  testing_array = np.loadtxt(CSVData, delimiter=",")
  img = []
  correct = 0

  for line in range(nb_pixels_images):
    img.append((testing_array[((nb_pixels_images+2)*indice_li)+line][indice_co]))
  img_to_pred = np.expand_dims(img, axis=0)
  img_to_pred = img_to_pred.reshape(-1,largeur_image, longueur_image, nb_channel)
  img_to_pred/=255.

  label = testing_array[((nb_pixels_images+2)*indice_li)+nb_pixels_images][indice_co]
  numero_cas_img_to_pred = testing_array[((nb_pixels_images+2)*indice_li)+nb_pixels_images+1][indice_co]
  predictions = model.predict(img_to_pred)
  classe = np.argmax(predictions, axis = 1)
  if(classe==label):
    correct=1
  else:
    plot(img_to_pred,label)
    correct=0

  print("\nPREDICTION DES CARACTERISTIQUES : ")
  print("numéro de l'image : ", label, "_", numero_cas_img_to_pred)
  print("caractéristiques à prédire : ", label)
  print("caractéristiques prédites : ", classe)
  if(label==classe):
    print("La prédiction est correcte !")
  else:
    print("Malheureusement la prédiction est incorrecte...")
  valeur_classe_predite = np.amax(predictions)

  return numero_cas_img_to_pred, label, classe, img_to_pred, correct

# ---------------------------------------------------------------------------------------------------------------------------------------------

def prediction_dataset(filename, testing_size, nb_pixels_images, model, n_classes):
  with open(filename) as file:
    total_good_predictions = 0
    good_predictions_per_classe = np.zeros(n_classes)
    bad_predictions_name = []
    bad_predictions_img = []
    realite = np.zeros(n_classes*testing_size)
    prediction = np.zeros(n_classes*testing_size)
    
    realite = realite.reshape(1,n_classes*testing_size)
    prediction = prediction.reshape(1,n_classes*testing_size)
    
    # parcours de l'ensemble de test
    for indice_li in range (n_classes):
      for indice_co in range(testing_size):
        numero_cas_img_to_pred, label, classe, img_to_pred, correct = prediction_1_image(filename, indice_li, indice_co, nb_pixels_images, model, n_classes)
        realite[0][indice_li*testing_size+indice_co] = float(label)
        prediction[0][indice_li*testing_size+indice_co] = float(classe)
        
        # évaluation des prédictions
        if(correct==1):
          total_good_predictions+=1
          good_predictions_per_classe[indice_li]+=1
        else:
          bad_case = str(label) + '_' + str(numero_cas_img_to_pred)
          bad_predictions_name.append(bad_case)
          bad_predictions_img.append(img_to_pred)

    # création matrice de confusion
    realite = realite.reshape(n_classes*testing_size,)
    prediction = prediction.reshape(n_classes*testing_size,)
    y_actu = pd.Series(realite, name='Reality')
    y_pred = pd.Series(prediction, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    mae = tf.keras.metrics.mean_absolute_error(realite, prediction).numpy()


    pourcentage_good_predictions = (total_good_predictions * 100) / (testing_size*n_classes)
    
    print("\nPREDICTION DU NOMBRE DES CARACTERISTIQUES A PARTIR DE L'ENSEMBLE DE TEST : ") 
    print("Le pourcentage de bonnes prédictions sur le testing set est de : ", pourcentage_good_predictions, "%. \n -------------------------------------------------------------------")
    for i in range(n_classes):
      print("Le pourcentage de bonnes prédictions pour la classe ", str(i+1), " est de : ", (good_predictions_per_classe[i]*100)/testing_size, "%.")    
    print(df_confusion)
    print(len(bad_predictions_name), "image(s) mal classée(s) :")
    for i in range(len(bad_predictions_name)):
      print("  ", bad_predictions_name[i])
      plot(bad_predictions_img[i],bad_predictions_name[i])
    print("MAE : ", mae)

  return good_predictions_per_classe

