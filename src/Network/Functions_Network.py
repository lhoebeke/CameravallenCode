import os
import sys
from ast import literal_eval
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import collections
from PIL import Image

import keras
import keras.backend as K

from keras.preprocessing.image import img_to_array, random_shift, random_rotation, random_shear, random_zoom #Additional data augmentation option. Add to DataGenerator.
from keras.applications.resnet50 import preprocess_input

from Preprocessing.Def_Functions import black_border

############################################################################################
def randomHorizontalFlip(image, p=0.5):
    """Do a random horizontal flip with probability p"""
    if np.random.random() < p:
        image = np.fliplr(image)
    return image

def randomVerticalFlip(image, p=0.5):
    """Do a random vertical flip with probability p"""
    if np.random.random() < p:
        image = np.flipud(image)
    return image

###########################################################################################
    
class DataGenerator(object):
    """Custom generator to train a keras model
    
    df: pandas DataFrame that maps 'image_name' to 'label' (these should be columns in the df)
    im_size (int): desired image size
    batch_size (int): batch_size for training
    shuffle (bool): shuffle the data at the start of each epoch
    mode ['train', 'test']:  At test mode, do not return labels; 
    augmentation (bool): on the fly augmentation/preprocessing
    
    Call .generate() to get the actual generator
    
    Code from Stijn Decubber
    """

    def __init__(self, df, dim_x, dim_y, dim_z, batch_size, shuffle=True, mode='train', augmentation=False):
        self.df = df  
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.augmentation = augmentation

    def _get_instance_indexes(self):
        """Fetch the indexes from the pandas df"""
        indexes = list(self.df.index)
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def _get_batch_images(self, indexes):
        """Return the images that correspond to the current batch"""
        batch_images = np.zeros((len(indexes), self.dim_x, self.dim_y, self.dim_z))
        
        datapath = resized_folder_path

        # Fill up container
        for i, ix in enumerate(indexes):

            image = Image.open(os.path.join(datapath, self.df['deployment'][ix] ,self.df['image_name'][ix]))
            image = image.crop(black_border(image))
            im = image.crop(self.df['box_standard'][ix])                   
            im = img_to_array(im)

            if self.augmentation:
                # Add augmentation or preprocessing here
                im = randomHorizontalFlip(im)
                #im = randomVerticalFlip(im)  
                #im = random_rotation(im, 25, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.)
                #im = random_shift(im, 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2,fill_mode='nearest', cval=0.)
                #im = random_shear(im, 0.2, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.)
                #im = random_zoom(im, (0.2,0.2), row_axis=0, col_axis=1, channel_axis=2,fill_mode='nearest', cval=0.)
            
            im = np.expand_dims(im, axis=0)
            im = preprocess_input(im, data_format=None, mode='tf')

            batch_images[i] = im

        return batch_images

    def _get_batch_labels(self, indexes):
        """Return the labels that correspond to the indices of the current batch"""
        if self.mode == 'test':
            return None
        else:
            return self.df['label'][indexes]

    def generate(self):
        """The actual generator"""

        classes = 17
        
        while True:
            indexes = self._get_instance_indexes()
            num_batches = int(np.ceil(len(self.df) / self.batch_size))
            for i in range(num_batches):
                if i == (num_batches - 1): # final batch can be smaller than actual batch size
                    batch_indexes = indexes[i * self.batch_size:]
                else:
                    batch_indexes = indexes[i * self.batch_size:(i + 1) * self.batch_size]

                X = self._get_batch_images(batch_indexes)
                y = self._get_batch_labels(batch_indexes)
                
                if self.mode == 'train':
                    y = keras.utils.to_categorical(y, num_classes=classes)
                    
                yield (X, y)
                
#####################################################################################
def split_mammals(x):
    
    """Function to add to a lambda layer to ensure that the probabilities of the child nodes add up to one."""
    
    Small_species = K.softmax(x[:,5:9])
    Large_species = K.softmax(x[:,9:])
    
    return K.concatenate([x[:,0:5], Small_species, Large_species], axis = 1)

######################################################################################
def convert_cond_probabilities(x):
    
    """Function to add to a lambda layer to convert the conditional probabilities to the output probabilities."""
    
    Blank = x[:,0]
    NotBlank = (1 - Blank)
    
    Animal = x[:,1] * NotBlank 
    NoAnimal = (1 - x[:,1]) * NotBlank
    
    Mammal = x[:,2] * Animal
    Bird = (1 - x[:,2]) * Animal
    
    Human = x[:,3] * NoAnimal 
    Pickup = (1 - x[:,3]) *NoAnimal
    
    Small = x[:,4] * Mammal
    Large = (1 - x[:,4]) * Mammal
      
    Mouse = x[:,5] * Small
    Squirrel = x[:,6] * Small
    Hare = x[:,7] * Small
    Hedgehog = x[:,8] * Small
    
    Ass = x[:,9] * Large
    Horse = x[:,10] * Large
    Fox = x[:,11] * Large
    Marten = x[:,12] * Large
    Cat = x[:,13] * Large
    Dog = x[:,14] * Large
    Mouflon = x[:,15] * Large
    Deer = x[:,16] * Large
    Boar = x[:,17] * Large
    
    return K.concatenate([K.expand_dims(Ass), K.expand_dims(Marten), K.expand_dims(Bird), 
                          K.expand_dims(Blank), K.expand_dims(Cat), K.expand_dims(Squirrel), 
                          K.expand_dims(Hare), K.expand_dims(Horse), K.expand_dims(Human), 
                          K.expand_dims(Mouse), K.expand_dims(Pickup), K.expand_dims(Fox), 
                          K.expand_dims(Dog), K.expand_dims(Mouflon), K.expand_dims(Hedgehog), 
                          K.expand_dims(Deer), K.expand_dims(Boar)], axis=1)

###################################################################################################
def load_train_val_test(FilePath):
    
    """Function to load the training, validation and test data from csv-files. 
    These file were created using the function split_train_val_test."""
    
    test = pd.read_csv(os.path.join(FilePath,'test_data.csv'),sep=';')
    train = pd.read_csv(os.path.join(FilePath,'train_data.csv'), sep=';')
    validation = pd.read_csv(os.path.join(FilePath,'validation_data.csv'), sep=';')
    
    #Make tuples readable
    test['box_standard'] = test['box_standard'].apply(literal_eval)
    train['box_standard'] = train['box_standard'].apply(literal_eval)
    validation['box_standard'] = validation['box_standard'].apply(literal_eval)

    
    return train, validation, test

######################################################################################################
def compute_weights(train, validation, test):
    
    """Function to compute the class weights. 
    This function return the class weights and a dictionary mapping the class names to the class labels."""
    
    df = pd.concat([train, test, validation])
    class_names = np.sort(df['ann_animal'].unique())
    classes = len(class_names)
    map_classes = dict(zip(class_names, range(classes)))
    
    counter_train=collections.Counter(train.ann_animal)

    ni = []
    for i in class_names:
        ni.append(counter_train[i])
        
    fi = np.divide(len(df),ni)
    wi = np.divide(fi,sum(fi))
    class_weights = {}
    for i in range(classes):
        class_weights[i] = wi[i]
    class_weights = dict(class_weights)
    
    return class_weights, map_classes

#######################################################################################################
def split_train_val_test(data, train_size, test_size, OutputPath):
    """Function to split the data into training, validation and test sequences and save the data. 
    The splitting is done in a stratified fashion, using the annotation."""
    
    data['annotation_literal'] = data['annotation'].apply(literal_eval)
    data = data.loc[data['annotation_literal'].str.len() == 1] #Select sequences with only one label
    data['box_standard'] = data['box_standard'].apply(literal_eval)
    data['ann_animal'] = ""  
    data['ann_animal'] = data['annotation_literal'].apply(group_birds)
    
    #Encode labels
    le = LabelEncoder()
    le.fit(data.ann_animal)
    data['label'] = le.transform(data.ann_animal)

    #Split data into training and 'testing' (validation + test) data
    data_sequences = data.loc[:,['sequence','ann_animal']].drop_duplicates(subset='sequence', keep='first', inplace=False)
    seq_train, seq_testing = train_test_split(data_sequences.sequence, test_size=(1-train_size), stratify = data_sequences.ann_animal)
    
    train = data.loc[data['sequence'].isin(seq_train)]
    testing = data.loc[data['sequence'].isin(seq_testing)]
    
    #Split 'testing' data into valdiation and test (validation + test) data
    test_sequences = testing.loc[:,['sequence','ann_animal']].drop_duplicates(subset='sequence', keep='first', inplace=False)
    seq_val, seq_test = train_test_split(test_sequences.sequence, test_size=test_size, stratify = test_sequences.ann_animal) 

    test = testing.loc[testing['sequence'].isin(seq_test)]
    validation = testing.loc[testing['sequence'].isin(seq_val)]
    
    #Save
    test.to_csv(os.path.join(OutputPath,'test_data.csv'), index=False, sep=';')
    train.to_csv(os.path.join(OutputPath,'train_data.csv'), index=False, sep=';')
    validation.to_csv(os.path.join(OutputPath,'validation_data.csv'), index=False, sep=';')
    
    return train, validation, test

##########################################################################################################
def group_birds(x):
    """Function to group all bird species into one class."""
    
    birds = ['Eurasian Blackbird', 'Song Thrush', 'House Sparrow', 'Common Pheasant', 
             'Great Tit', 'Short-toed Treecreeper', 'Greylag Goose', 'Carrion Crow', 
             'Great Spotted Woodpecker', 'Eurasian Jay']

    if x[0] in birds:
        return 'Bird'
    else:
        return x[0]
    
########################################################################################################
def split_train_val_test_bottleneck(bottleneck_data, bottleneck, train_size, test_size, OutputPath):
    
    """Function to split the bottleneck features and data into training, validation and test sequences and save the data. 
    The splitting is done in a stratified fashion, using the annotation."""
    
    #Split data into training and 'testing' (validation + test) data
    bottleneck_sequences = bottleneck_data.loc[:,['sequence','ann_animal']].drop_duplicates(subset='sequence', keep='first', inplace=False)
    seq_train, seq_testing = train_test_split(bottleneck_sequences.sequence, test_size=(1-train_size), stratify = bottleneck_sequences.ann_animal)
    
    train = bottleneck_data.loc[bottleneck_data['sequence'].isin(seq_train)]
    testing = bottleneck_data.loc[bottleneck_data['sequence'].isin(seq_testing)]
    
    #Split 'testing' data into valdiation and test (validation + test) data
    test_sequences = testing.loc[:,['sequence','ann_animal']].drop_duplicates(subset='sequence', keep='first', inplace=False)
    seq_val, seq_test = train_test_split(test_sequences.sequence, test_size=test_size, stratify = test_sequences.ann_animal) 

    test = bottleneck_data.loc[bottleneck_data['sequence'].isin(seq_test)]
    validation = bottleneck_data.loc[bottleneck_data['sequence'].isin(seq_val)]
    
    bottleneck_features_train = bottleneck[bottleneck_data['sequence'].isin(seq_train),:,:,:]
    bottleneck_features_test = bottleneck[bottleneck_data['sequence'].isin(seq_test),:,:,:]
    bottleneck_features_validation = bottleneck[bottleneck_data['sequence'].isin(seq_val),:,:,:]
    
    np.save(os.path.join(OutputPath, 'bottleneck_features_train.npy'), bottleneck_features_train)
    np.save(os.path.join(OutputPath, 'bottleneck_features_test.npy'), bottleneck_features_test)
    np.save(os.path.join(OutputPath, 'bottleneck_features_validation.npy'), bottleneck_features_validation)

    test.to_csv(os.path.join(OutputPath,'test_data.csv'), index=False, sep=';')
    train.to_csv(os.path.join(OutputPath,'train_data.csv'), index=False, sep=';')
    validation.to_csv(os.path.join(OutputPath,'validation_data.csv'), index=False, sep=';')
    
    return train, validation, test, bottleneck_features_train, bottleneck_features_validation, bottleneck_features_test

###########################################################################################################################
