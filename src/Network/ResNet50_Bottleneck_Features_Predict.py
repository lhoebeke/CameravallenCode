import os
from ast import literal_eval
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from keras.applications import ResNet50

from network.functions_network import group_birds, DataGenerator


def extract_bottleneck_features(preprocessing_output_path, bottleneck_features_output_path, resized_folder_path):
    
    """Function to extract the bottleneck features using the pretrained network ResNet50.
    
    Parameters
    ----------
    preprocessing_output_path : string (filepath)
        path to folder with the preprocessing output files
    bottleneck_features_output_path : string (filepath)
        path to folder where the bottleneck features will be saved
    resized_folder_path : string (filepath)
        path to folder containing the resized images

 
    Returns
    -------
    Following files will be saved on disk:
        - bottleneck_features.npy : numpy array containing the bottleneck features
        - bottleneck_data.csv : data corresponding to the extracted bottleneck features
    """
    
    #Import preprocessing data
    data = pd.read_csv(os.path.join(preprocessing_output_path,'boxes_preprocessing_single.csv'), sep=';')
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
    
    #Save data corresponding to the bottleneck features
    data.to_csv(os.path.join(bottleneck_features_output_path,'bottleneck_data.csv'), index=False, sep=';')
    
    #Image size
    dim_x = 270
    dim_y = 480
    dim_z = 3
    
    #Parameters
    batch_size = 128
    steps_per_epoch = np.ceil(len(data) / batch_size)
    
    #Build the ResNet50 network
    model = ResNet50(include_top=False, weights='imagenet')
    
    #Create generator
    generator = DataGenerator(data, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, 
                              batch_size=batch_size, augmentation=False, shuffle=False, mode='train').generate()
    
    #Extract and save bottleneck features
    bottleneck_features = model.predict_generator(generator, steps_per_epoch)
    np.save(os.path.join(bottleneck_features_output_path, 'bottleneck_features.npy'), bottleneck_features)
    
if __name__ == '__main__':
    extract_bottleneck_features()