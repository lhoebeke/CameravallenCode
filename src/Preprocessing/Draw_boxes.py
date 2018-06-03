import os
from ast import literal_eval
import pandas as pd
from PIL import Image, ImageDraw

from Preprocessing.Def_Functions import black_border


def draw_boxes(preprocessing_output_path, resized_folder_path, draw_output_path):
    
    """ Function to indicate the regions of interest on the camera trap images.

    Parameters
    ----------
    preprocessing_output_path : string (filepath)
        path to folder containing the preprocessing output files
    resized_folder_path : string (filepath)
        path to folder with resized images
    draw_output_path : string (filepath)
        path to folder where the images will be saved
    """
    
    #Import preprocessing output
    list_preprocessing = pd.read_csv(os.path.join(preprocessing_output_path,'boxes_preprocessing.csv'), sep=';')
    
    #Crop images and/or draw boxes on camera trap image
    for row in list_preprocessing.itertuples():
        image = Image.open(os.path.join(resized_folder_path, row.deployment, row.image_name))
        image = image.crop(black_border(image))
              
        #Draw boxes
        im_box = ImageDraw.Draw(image)
        for box in literal_eval(row.box_standard):
            if len(box) != 0:
                im_box.rectangle(box, fill=None, outline = 'red')   
        name = row.image_name[:-4] + '_box.jpg'
        
        if not os.path.exists(os.path.join(draw_output_path, row.deployment)):
            os.makedirs(os.path.join(draw_output_path,row.deployment))
                
        image.save(os.path.join(draw_output_path, row.deployment, name))

if __name__ == '__main__':
    draw_boxes()