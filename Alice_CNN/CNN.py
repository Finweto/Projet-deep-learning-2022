# -*- coding: utf-8 -*-
"""
CNN.py
# estimation des caractéristiques de plantes : images de plantes -> CNN -> caractéristiques
# 02/12/2022
"""

print("\n-------------\nBONJOUR, JE SUIS GULDURNet ! \nJE VAIS ESTIMER LES CARACTERISTIQUES DE PLANTES A PARTIR DE LEUR PHOTO \nARCHITECTURE UTILISEE : CNN.\n-------------\n")


# IMPORTS -------------------------------------------------------------------------------------------------------
print("\nIMPORTS ...")
import CNN_bibli as bibli
import random


# DATASET -------------------------------------------------------------------------------------------------------
print("\nLIENS AVEC LES DATASETS")
TRAINING_FILE = "/cea/home/b4/cameijoa/Desktop/cameijoa/python/DL/DATASET/training.csv"
TESTING_FILE = "/cea/home/b4/cameijoa/Desktop/cameijoa/python/DL/DATASET/testing.csv"


# VARIABLES GLOBALES -------------------------------------------------------------------------------------------------------
# nombres de colonnes (images) dans les datasets
training_size = 20 
testing_size = 10
# nombre de lignes 
ligne_training_csv = ???
# autres informations
nb_pixels_images = ???
n_classes = 11


# PARSING -------------------------------------------------------------------------------------------------------
print("\nPARSING ... ")
training_images, training_labels = bibli.parse_data_from_input(TRAINING_FILE, training_size, ligne_training_csv,  validation_size, nb_pixels_images, n_classes)
print(f"Training images has shape: {training_images.shape} and dtype: {training_images.dtype}")
print(f"Training labels has shape: {training_labels.shape} and dtype: {training_labels.dtype}")


# DATA AUGMENTATION -------------------------------------------------------------------------------------------------------
print("\nDATA AUGMENTATION")
train_generator = bibli.simple_train_val_generators(training_images, training_labels, training_size, n_classes)
print(f"Images of training generator have shape: {train_generator.x.shape}")
print(f"Labels of training generator have shape: {train_generator.y.shape}")


# RESEAU -------------------------------------------------------------------------------------------------------
print("\nCNN")
model = bibli.run_model(train_generator, n_classes)


# PREDICTIONS -------------------------------------------------------------------------------------------------------
# prédiction sur 1 cas de l'ensemble de test 
cas, label, classe, img, correct = bibli.prediction_1_image(TESTING_FILE, random.randint(0,(n_classes-1)), random.randint(0,(testing_size-1)), nb_pixels_images, model, n_classes)

# prédiction sur l'ensemble des cas de l'ensemble de test
tab_good_pred_per_classe = bibli.prediction_dataset(TESTING_FILE, testing_size, nb_pixels_images, model, n_classes)


    
