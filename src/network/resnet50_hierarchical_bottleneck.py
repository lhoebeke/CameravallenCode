import os
import time 
import pickle
import pandas as pd
import numpy as np
from yaml import load

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Lambda

from network.functions_network import (load_train_val_test,
                                       compute_weights,
                                       split_mammals,
                                       convert_cond_probabilities)
  
# Load configuration file
with open("config.yml") as yaml_config:
    config = load(yaml_config)
        
start_time = time.time()

#Load training, validation and test data
train, validation, test = load_train_val_test(config['bottleneck_features_output_path'])

#Load training, validation and test bottleneck features
bottleneck_features_train = np.load(os.path.join(config['bottleneck_features_output_path'], 'bottleneck_features_train.npy'))
bottleneck_features_test = np.load(os.path.join(config['bottleneck_features_output_path'], 'bottleneck_features_test.npy'))
bottleneck_features_validation = np.load(os.path.join(config['bottleneck_features_output_path'], 'bottleneck_features_validation.npy'))

#Compute weights
weights, map_classes = compute_weights(train, validation, test)
classes = len(map_classes)

#Image size
dim_x = 270
dim_y = 480
dim_z = 3

#Parameters
batch_size = 128
ep = 3000

steps_per_epoch = np.ceil(len(train) / batch_size)
val_steps_per_epoch = np.ceil(len(validation) / batch_size)


#Convert labels to categorical labels
train_labels = keras.utils.to_categorical(np.array(train['label']),num_classes=classes)
validation_labels = keras.utils.to_categorical(np.array(validation['label']), num_classes=classes)

#Load top model

#Number of output classes in the classification tree
cond_classes = 5+4+9

top_model = Sequential()
top_model.add(Flatten(input_shape=(1, 2, 2048)))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.50))
top_model.add(Dense(cond_classes, activation='sigmoid'))
top_model.add(Lambda(split_mammals,name='cond_layer'))
top_model.add(Lambda(convert_cond_probabilities))

#Compile model
opt = optimizers.Adam(clipvalue=0.01)
top_model.compile(optimizer=opt,
              loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#Train model
history = top_model.fit(bottleneck_features_train, train_labels,
          epochs=ep,
          batch_size=batch_size,
          validation_data=(bottleneck_features_validation, validation_labels),class_weight=weights)

#Save model, weights and history
top_model.save_weights(os.path.join(config['weight_path'], 'nesnet_bottleneck_weights.h5'))
top_model.save(os.path.join(config['weight_path'], 'resnet50_bottleneck_model.h5'))
with open(os.path.join(config['weight_path'],'train_history_dict'), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

#Predict and save pedictions for validation and test data
predictions = top_model.predict(bottleneck_features_validation)
pd.DataFrame(predictions).to_csv(os.path.join(config['predictions_output_path'],'validation_predictions.csv'), index=False, sep=';')
predictions_test = top_model.predict(bottleneck_features_test)
pd.DataFrame(predictions_test).to_csv(os.path.join(config['predictions_output_path'],'test_predictions.csv'), index=False, sep=';')

#Print time
print("--- %s minutes ---" % format((time.time() - start_time)/60, '.2f'))

