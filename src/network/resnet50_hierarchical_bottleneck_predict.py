import os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import (Dropout, Flatten, Dense, 
                          Lambda)

from network.functions_network import split_mammals, convert_cond_probabilities

def hierarchical_bottleneck_predict(bottleneck_features_output_path, weight_path, predictions_output_path):
    
    """Function to predict the labels using the extracted bottleneck features.
    
    Parameters
    ----------
    bottleneck_features_output_path : string (filepath)
        path to folder containing the bottleneck features
    weight_path : string (filepath)
        path to folder containing weights of the top model
    predictions_output_path : string (filepath)
        path to folder where the predictions will be saved
    
    
    Returns
    -------
    Following files will be saved on disk:
        - predictions.csv : file containing the output probabilities for the images
        
    """
    
    #Load bottleneck features and data
    bottleneck_features = np.load(os.path.join(bottleneck_features_output_path, 'bottleneck_features.npy'))
    
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
    top_model.load_weights(os.path.join(weight_path,'resnet_bottleneck_weights.h5'), by_name=False)
    
    #Predict and save predictions
    predictions = top_model.predict(bottleneck_features)
    pd.DataFrame(predictions).to_csv(os.path.join(predictions_output_path,'predictions.csv'), index=False, sep=';')


if __name__ == '__main__':
    hierarchical_bottleneck_predict()