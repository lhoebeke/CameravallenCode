import pandas as pd
import os

from network.functions_output import (hierarchical_predictions, 
                                      bottom_hierarchical_prediction,
                                      top_predictions)

def hierarchical_predictions_sequences(predictions_output_path, bottleneck_features_output_path):
    
    """This function determines the hierchical classification of the sequences, using the top-k method.
    
    Parameters
    ----------
    predictions_output_path : string (filepath)
        path to folder containing the predictions of the network
    bottleneck_features_output_path : string (filepath)
        path to folder containing the data of the bottleneck features

    Returns
    -------
    Following files will be saved on disk:
        - hierarchical_predictions_sequences.csv : hierarchical labels of the sequences
    """

    # Import data
    data_bottleneck = pd.read_csv(os.path.join(bottleneck_features_output_path,'bottleneck_data.csv'), sep = ';')
    predictions = pd.read_csv(os.path.join(predictions_output_path, 'predictions.csv'), sep = ';')
    data = pd.concat([data_bottleneck, predictions], axis=1)
    
    # Hierarchical classification images
    hierarchy = hierarchical_predictions(data)
    data_hierarchy = pd.concat([data_bottleneck, hierarchy], axis=1)
    
    # Hierarchical classification sequences using the top-k method
    pred_top = top_predictions(data, data_hierarchy)
 
    #Final prediction for every sequence
    pred_top['final_prediction'] = pred_top.apply(bottom_hierarchical_prediction, axis=1)
    pred_top.to_csv(os.path.join(predictions_output_path, 'hierarchical_predictions_sequences.csv'), index=False, sep=';')


if __name__ == '__main__':
    hierarchical_predictions_sequences()