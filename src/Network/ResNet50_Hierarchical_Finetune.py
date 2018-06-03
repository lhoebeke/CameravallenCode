import os
import time 
import pickle

import pandas as pd
import numpy as np
from yaml import load

from keras.applications import ResNet50
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Lambda

from Network.Functions_Network import (load_train_val_test,
                                       compute_weights,
                                       split_mammals,
                                       convert_cond_probabilities,
                                       DataGenerator)
# Load configuration file
with open("config.yml") as yaml_config:
    config = load(yaml_config)
    
#Start timer
start_time = time.time()

#Import training, validation and test data.
train, validation, test = load_train_val_test(config['bottleneck_features_output_path'])

#Compute weights
class_weights, map_classes = compute_weights(train, validation, test)
classes = len(map_classes)

#Image dimensions
dim_x = 270
dim_y = 480
dim_z = 3

#Training parameters
batch_size = 64*2
ep = 100

train_steps_per_epoch = np.ceil(len(train) / batch_size)
val_steps_per_epoch = np.ceil(len(validation) / batch_size)
test_steps_per_epoch = np.ceil(len(test) / batch_size)


#Build model

#Number of output classes in the classification tree
cond_classes = 5+4+9

#Trained top
top_model = Sequential()
top_model.add(Flatten(input_shape=(1, 2, 2048)))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.50))
top_model.add(Dense(cond_classes, activation='sigmoid'))
top_model.add(Lambda(split_mammals,name='cond_layer'))
top_model.add(Lambda(convert_cond_probabilities))
top_model.load_weights(os.path.join(config['weight_path'],'ResNet_bottleneck_weights.h5'), by_name=False)

conv_base = ResNet50(include_top=False, weights='imagenet', input_shape=(270, 480, 3)) #Pretrained base
conv_base.trainable = True #Unfreeze convolutional base to finetune

model = Sequential()
model.add(conv_base)
model.add(top_model)


#Select trainable layers
layer_names = [] 
for layer in conv_base.layers: 
    layer_names.append(layer.name)

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

#Compile model       
opt = optimizers.Adam(clipvalue=0.01)
model.compile(opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])


#Create generator for training, validation and test data
train_generator = DataGenerator(df=train, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, 
                          batch_size=batch_size, augmentation=False, shuffle=True, mode='train').generate()

val_generator = DataGenerator(df=validation, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, 
                          batch_size=batch_size, augmentation=False, shuffle=False, mode='train').generate()

test_generator = DataGenerator(df=test, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, 
                          batch_size=batch_size, augmentation=False, shuffle=False, mode='train').generate()

#Train model 
history = model.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=ep, verbose=1, 
                              validation_data=val_generator, validation_steps=val_steps_per_epoch, class_weight=class_weights)

#Safe history, model and weights.
model.save_weights(os.path.join(config['weight_path'], 'ResNet50_finetune_weights.h5'))
model.save(os.path.join(config['weight_path'], 'ResNet50_finetune_model.h5'))
with open(os.path.join(config['weight_path'],'trainHistoryDict'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

#Predict and save prediction validation and test data
predictions_val = model.predict_generator(val_generator, val_steps_per_epoch, verbose=1)
pd.DataFrame(predictions_val).to_csv(os.path.join(config['predictions_output_path'],'validation_predictions.csv'), index=False, sep=';')

predictions_test = model.predict_generator(test_generator, test_steps_per_epoch, verbose=1)
pd.DataFrame(predictions_test).to_csv(os.path.join(config['predictions_output_path'],'test_predictions.csv'), index=False, sep=';')

#Print time
print("--- %s minutes ---" % format((time.time() - start_time)/60, '.2f'))

