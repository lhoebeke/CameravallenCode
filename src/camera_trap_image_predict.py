
import click
from yaml import load
import os

from preprocessing.resize_images import resize_images
from preprocessing.preprocessing import preprocessing
from network.resnet50_bottleneck_features_predict import extract_bottleneck_features
from network.resNet50_hierarchical_bottleneck_predict import hierarchical_bottleneck_predict
from network.hierachical_processing_predictions import hierarchical_predictions_sequences

@click.command()
@click.argument("configuration_file")
def main(configuration_file):
    
    """Main function to hierarchically classify new camera trap images.
    
    Parameters
    ----------
    configuration_file : string 
        file containing the all paths
    """
    
    # Load configuration file
    with open("config.yml") as yaml_config:
        config = load(yaml_config)
        
    # Check if all folders exist and create folder if needed
    for path in config:
        if not os.path.exists(config[path]):
            os.makedirs(config[path])
    
    
    # Step 1: resize images
    ########################
    # Input: images and Agouti export file (observations)
    # Output: resized images in similar folder structure as original images

    resize_images(config["general_folder_path"], 
                  config["resized_folder_path"])
    
    # Step 2: preprocess images
    ###########################
    # Input: resized images and Agouti export (observations + assets + pickupsetup)
    # Output: file containing coordinates of the regions of interest
    
    preprocessing(config["general_folder_path"], 
                  config["resized_folder_path"], 
                  config["preprocessing_output_path"])
    
    # Step 3: extract bottleneck features using the pretrained network ResNet50
    ############################################################################
    # Input: resized images and preprocessing output containing the coordinates of the boxes
    # Output: bottleneck features of all images
    
    extract_bottleneck_features(config["preprocessing_output_path"], 
                                config["bottleneck_features_output_path"], 
                                config["resized_folder_path"])
    
    # Step 4 : run top model to classify the new images
    ##################################################
    # Input: extracted bottleneck features
    # Ouput: predicted probabilities
    
    hierarchical_bottleneck_predict(config["bottleneck_features_output_path"], 
                                    config["weight_path"], 
                                    config["predictions_output_path"])
    
    # Step 5 : convert output probabilities to hierarchical classification
    ######################################################################
    # Input: predicted probabilities
    # Output: hierarchical classification of the sequences
    
    hierarchical_predictions_sequences( config["predictions_output_path"], 
                                       config["bottleneck_features_output_path"])
    

if __name__ == '__main__':
    main()