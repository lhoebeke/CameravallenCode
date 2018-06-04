import os
from ast import literal_eval
import pandas as pd
import numpy as np
from yaml import load

from sklearn.preprocessing import LabelEncoder
from keras.applications import ResNet50

from network.functions_network import (group_birds, 
                                       DataGenerator, 
                                       split_train_val_test_bottleneck)

# Load configuration file
with open("config.yml") as yaml_config:
    config = load(yaml_config)
    
#Import preprocessing data
data = pd.read_csv(os.path.join(config['preprocessing_output_path'],'boxes_preprocessing_single.csv'), sep=';')
data['annotation_literal'] = data['annotation'].apply(literal_eval)
data['box_standard'] = data['box_standard'].apply(literal_eval)

#Select sequences with only one label
data = data.loc[data['annotation_literal'].str.len() == 1] 

#Group all bird species into one class
data['ann_animal'] = ""
data['ann_animal'] = data['annotation_literal'].apply(group_birds)

#Encode labels
le = LabelEncoder()
le.fit(data.ann_animal)
data['label'] = le.transform(data.ann_animal)
map_classes = dict(zip(le.classes_, range(len(le.classes_))))
classes = len(map_classes)

#Save data corresponding to the bottleneck features
data.to_csv(os.path.join(config['bottleneck_features_output_path'],'bottleneck_data.csv'), index=False, sep=';')

#Image size
dim_x = 270
dim_y = 480
dim_z = 3

#Parameters
batch_size = 100
steps_per_epoch = np.ceil(len(data) / batch_size)

#Build the ResNet50 network
model = ResNet50(include_top=False, weights='imagenet')

#Create generator
generator = DataGenerator(data, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, 
                          batch_size=batch_size, augmentation=False, shuffle=False, mode='train').generate()

#Extract and save bottleneck features
bottleneck_features = model.predict_generator(generator, steps_per_epoch)
np.save(os.path.join(config['bottleneck_features_output_path'], 'bottleneck_features.npy'), bottleneck_features)

#When training:
#Split bottleneck features and data into training, validation and test data
train, validation, test, bottleneck_features_train, bottleneck_features_validation, bottleneck_features_test = split_train_val_test_bottleneck(data, bottleneck_features, 0.5, 0.25, config['bottleneck_features_output_path'])