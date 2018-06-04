import glob 
import os
import pandas as pd
from PIL import Image

def resize_images(general_folder_path, resized_folder_path):
    
    """ Function to resize the original camera trap images.

    Parameters
    ----------
    general_folder_path : string (filepath)
        path to folder containing the raw data (the images and the Agouti export files)
    resized_folder_path : string (filepath)
        path to folder where the resized images will be saved
    
    """
    
    #Load Agouti export with observation
    observations = pd.read_csv(os.path.join(general_folder_path, 'observations.csv'))
    
    
    #Loop over every deployment
    deployments = observations.deploymentID.unique()
    
    for folder in os.listdir(general_folder_path):
        imageFolderPath = os.path.join(general_folder_path, folder)
        
        #Check if it is a folder, not a file and if deployment is annotated
        if os.path.isdir(imageFolderPath) and folder in deployments and folder not in os.listdir(resized_folder_path):
            imagePath = glob.glob(imageFolderPath + '/*.JPG')
            
            if not os.path.exists(os.path.join(resized_folder_path, folder)):
                os.makedirs(os.path.join(resized_folder_path, folder))
                  
            #Import all images
            for img in imagePath:
                im = Image.open(img)
                size =im.size   # get the size of the input image
                RATIO = 0.5  # reduced the size to 50% of the input image
                reduced_size = int(size[0] * RATIO), int(size[1] * RATIO)     
                
                im_resized = im.resize(reduced_size)
                image_name = (im.filename).split('\\')[-1]
                im_resized.save(os.path.join(resized_folder_path,folder,image_name))

if __name__ == '__main__':
    resize_images()