# Camera traps recognition and identification

This repository contains the code from my master thesis 'Automated recognition of people and identification of animal species in camera trap images'. In this master thesis, existing manually labelled images from a camera trap study conducted by the Research Institute for Nature and Forest (INBO) in collaboration with Hasselt University are used to train a convolutional neural network to hierarchically classify camera trap images. The data used for the training has been annotated using [Agouti](https://www.agouti.eu/), a platform for managing wildlife camera-trapping projects.

You can either use the trained network to classify new sequences or retrain the network itself.

## Setting up environment

The code is written in Python and the working environment with the required packages can be set up using the `environment.yml` file. To prepare and setup the environment with the command line (see also [here](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)):

```
conda env create -f environment.yml
```
or [use the Anaconda navigator functionality](https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments#importing-an-environment) to load the `environment.yml` file.

## A. Use the network to hierarchically classify new sequences

### Configurations

All input and output paths are listed in the file `config.yml`. You can either use the predefined paths or change them according to your own data structure. This repo assumes the following data structure:

```
data 
  ├─ raw  # Orginal camera trap images and the Agouti export files
      ├─ observations.csv
      ├─ assets.csv
      ├─ pickup_setup.csv
      └─ deployment_hash_1
           ├─ photo_1
           ├─ ...
      ├─ ...
      └─ deployment_hash_n
           ├─ photo_1
           ├─ ...
  ├─ interim
      ├─ resized
      ├─ preprocessing
      ├─ crop
      ├─ draw
      └─ bottleneck_features
  ├─ processed
      └─ hierarchical_predictions_sequences
```
__Remark:__ The interim subfolders are automatically when running the tutorial.

### Tutorial

The jupyter notebook `Tutorial_predictions.ipynb` contains a short tutorial on how you can use the network to classify new sequences.

### Command line

The command line version combines all 5 steps in a single pipeline. You can run `Camera_trap_image_predict.py` using the command line:

```
python Camera_trap_image_predict.py --config.yml
```

### To hierarchically classify new sequences, the following steps are executed:

**1. Resize the original camera trap images** <br>
The camera trap images are resized to 50% of their original size. This strongly decreases the computational time, while the performance remains the same.

**2. Preprocess the resized images** <br>
During the preprocessing, the regions of interest in the images are determined. All images of a sequence are used to construct a background image. Subsequently, the regions of interest in a camera trap image are determined by computing the difference between this background image and the camera trap image.

**3. Extract the bottleneck features** <br>
The pretrained convolutional neural network ResNet50 is used to convert the images to bottleneck features.

**4. Run the top model to classify the new images** <br>
The extracted bottleneck features are fed to the new top model to predict the labels of the new images.

**5. Convert the output probabilities to a hierarchical classification** <br>
The predictions of the individual image regions are aggregated to a hierarchical prediction for every sequence.

**Additional steps** <br>
These steps are optional and not required to classify the images.

- **Crop image or indicate regions of interest** <br>
To see the result of the preprocessing the camera trap images can be cropped or the regions of interest can be indicated on the camera trap images .

- **Object localization** <br>
The class activation maps can be used to localize the objects in the cropped camera trap images. <br>
Since this step uses the cropped images, make sure to first run the optional cropping step mentioned above.

## B. Retrain the convolutional neural network
This repository also contains some scripts that can be used to retrain the convolutional network. These scripts also use the configuration file `config.yml`.

**1. Resize the original camera trap images** <br>

**2. Preprocess the resized images** <br>

**3. Retrain the model** <br>

**- Train model without data augmentation** <br>
To train the model without data augmentation, first run `ResNet50_Bottleneck_Features.py` to extract the bottleneck features. Subsequently, run `ResNet50_Hierarchical_Bottleneck` to train the model.

**- Train model with data augmentation** <br>
To train the model with data augmentation, run `ResNet50_Hierarchical_Augmentation`. You can change the augmentation options in   `DataGenerator` in `Functions_Network.py`.

**- Fine-tuning** <br>
To fine-tune the model, run `ResNet50_Hierarchical_Finetune`. You can either use the top weights provided in this repository or retrain the top before fine-tuning.

**4. Evaluate the performance of the retrained network** <br>
`Hierachical_output_processing_training.py`can be used to evaluate the performance of the network. This script allows you to aggregate the predictions of the individual images to a hierarchical label for the sequence. Code to plot the training history and confusion matrices is also provided.
