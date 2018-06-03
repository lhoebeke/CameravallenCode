#bron: https://keras.io/applications/
import os
import time 
import pickle
import pandas as pd
import numpy as np
from yaml import load

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Lambda
from keras.applications import ResNet50

from Network.Functions_Network import (split_train_val_test,
                                       load_train_val_test,
                                       compute_weights,
                                       split_mammals,
                                       convert_cond_probabilities,
                                       DataGenerator)

# Load configuration file
with open("config.yml") as yaml_config:
    config = load(yaml_config)
    
start_time = time.time()

#Load preprocessing data
data = pd.read_csv(os.path.join(config['preprocessing_output_path'],'boxes_preprocessing_single.csv'), sep=';') 

#Split training, validation and test data
train, validation, test = split_train_val_test(data, 0.5, 0.25, config['preprocessing_output_path'])

#Or load train, validdation and test data
#train, validation, test = load_train_val_test(config['preprocessing_output_path'])

#Compute weights
class_weights, map_classes = compute_weights(train, validation, test)
classes = len(map_classes)

#Image dimensions
dim_x = 270
dim_y = 480
dim_z = 3

#Parameters
batch_size = 64
ep = 200

train_steps_per_epoch = np.ceil(len(train) / batch_size)
val_steps_per_epoch = np.ceil(len(validation) / batch_size)
test_steps_per_epoch = np.ceil(len(test) / batch_size)


#Build model

#Number of output classes in the classification tree
cond_classes = 5+4+9
conv_base = ResNet50(include_top=False, weights='imagenet', input_shape=(270, 480, 3))

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(cond_classes, activation='sigmoid'))
model.add(Lambda(split_mammals,name='cond_layer'))
model.add(Lambda(convert_cond_probabilities))

conv_base.trainable = False #Freeze convolutional base
    
#Compile model
opt = optimizers.Adam(clipvalue=0.01)
model.compile(opt, loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'])

#Create generator for training, validation and test data
train_generator = DataGenerator(df=train, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, 
                          batch_size=batch_size, augmentation=True, shuffle=True, mode='train').generate() 
#Go to code DataGenerator to add different data augmentation options.

val_generator = DataGenerator(df=validation, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, 
                          batch_size=batch_size, augmentation=False, shuffle=False, mode='train').generate()

test_generator = DataGenerator(df=test, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, 
                          batch_size=batch_size, augmentation=False, shuffle=False, mode='train').generate()

#Train model with augmentation
history = model.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=ep, verbose=1, 
                              validation_data=test_generator, validation_steps=val_steps_per_epoch, class_weight=class_weights)

#Save model, weights and history
model.save_weights(os.path.join(config['weight_path'], 'ResNet50_augmentation_weights.h5'))
model.save(os.path.join(config['weight_path'], 'ResNet50_augmentation_model.h5'))
with open(os.path.join(config['weight_path'],'trainHistoryDict'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

#Predict and save prediction validation and test data
predictions_val = model.predict_generator(val_generator, val_steps_per_epoch, verbose=1)
pd.DataFrame(predictions_val).to_csv(os.path.join(config['predictions_output_path'],'validation_predictions.csv'), index=False, sep=';')

predictions_test = model.predict_generator(test_generator, test_steps_per_epoch, verbose=1)
pd.DataFrame(predictions_test).to_csv(os.path.join(config['predictions_output_path'],'test_predictions.csv'), index=False, sep=';')

print("--- %s minutes ---" % format((time.time() - start_time)/60, '.2f'))

