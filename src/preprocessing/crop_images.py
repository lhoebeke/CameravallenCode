import os
from ast import literal_eval
import pandas as pd
from PIL import Image

from preprocessing.def_functions import black_border

def crop_images(preprocessing_output_path, resized_folder_path, crop_output_path):
    
    """ Function to extract the regions of interest from the camera trap images.

    Parameters
    ----------
    preprocessing_output_path : string (filepath)
        path to folder containing the preprocessing output files
    resized_folder_path : string (filepath)
        path to folder with resized images
    crop_output_path : string (filepath)
        path to folder where the cropped images will be saved
    """
    
    #Import preprocessing output
    list_preprocessing = pd.read_csv(os.path.join(preprocessing_output_path,'boxes_preprocessing.csv'), sep=';')
    
    #Crop images and/or draw boxes on camera trap image
    for row in list_preprocessing.itertuples():
        image = Image.open(os.path.join(resized_folder_path, row.deployment, row.image_name))
        image = image.crop(black_border(image))
        
        #Crop images
        if row.image_type == 'grey':
            image = image.convert('L')
            
        for box in literal_eval(row.box_standard):
            i = 0
            if len(box) != 0:
                i += 1
                region =image.crop(box)
                name = row.image_name[:-4] + '_crop_' + str(i+1)+ '.jpg'
                region.save(os.path.join(crop_output_path, name))

if __name__ == '__main__':
    crop_images()