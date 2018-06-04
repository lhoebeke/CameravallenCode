import os
import numpy as np
import pandas as pd

from PIL import Image, ImageChops, ImageFilter
from skimage import measure
from skimage.filters import roberts
from scipy import ndimage as ndi
import cv2

from preprocessing.def_functions import remove_dup_columns, black_border, standard_box, size_box, devide_box


def preprocessing(general_folder_path, resized_folder_path, preprocessing_output_path):
    
    """This function determines the regions of interest in the camera trap images.
    
    Parameters
    ----------
    general_folder_path : string (filepath)
        path to folder containing the Agouti export files
    resized_folder_path : string (filepath)
        path to folder containing the resized images
    preprocessing_output_path : string (filepath)
        path to folder where the output files are saved
    
    
    Returns
    -------
    Following files will be saved on disk:
        - data_start_preprocessing.csv : combinations of the Agouti export (assets, observations and SetupPickup)
        - data_preprocessing.csv : total output of the preprocessing
        - boxes_preprocessing.csv : file containing the coordinates of the regions of interest for every image
        - boxes_preprocessing_single.csv : similar to boxes_preprocessing.csv, but now each row corresponds to 
            a region of interest, while in boxes_preprocessing.csv the rows correspond to the images
    """

    #Resized images
    RATIO = 0.5
    
    #Import Agouti export files
    observations = pd.read_csv(os.path.join(general_folder_path, 'observations.csv'))
    assets = pd.read_csv(os.path.join(general_folder_path, 'assets.csv'), low_memory = False)
    setup = pd.read_csv(os.path.join(general_folder_path, 'pickup_setup.csv'), sep = ';')
    
    #Combine annotations for sequences with multiple annotations
    list_columns = ['animalCount','animalTaxonID','animalIsDomesticated','animalScientificName','animalVernacularName','animalSex','animalAge', 'animalBehavior', 'deploymentID']
    observations_unique = pd.DataFrame()   
    for name in list_columns:
        column_unique = observations.groupby('sequenceID')[name].apply(list).reset_index()
        observations_unique = pd.concat([observations_unique, column_unique], axis=1)
    observations_unique = remove_dup_columns(observations_unique)
    
    #Join annotations and pickup-setup data
    ann = assets.set_index('sequence').join(observations_unique.set_index('sequenceID'))
    data = ann.join(setup.set_index('sequenceId'))
    data.reset_index(level=0, inplace=True)
    data.rename(columns={'index': 'sequenceID'}, inplace=True)
    data = data.drop([ 'id','type','originalFilename','destination','directory','exiftoolData','order',
                      'createdAt','isFavourite','observations','isTimeLapse','deployment'], axis=1)
        
    #Combine deploymentID of observation and pickup-setup into one column    
    data['deployment'] = ""
    for i, row in data.iterrows():
        if isinstance(row.deploymentID, list):
            row.deployment = row.deploymentID[0]
        else:
            row.deployment = row.deploymentId
    data = data.drop(['deploymentId', 'deploymentID'], axis=1)       
            
    #Combine annotations from observations and pickup-setup into one column     
    data['Annotation'] = ""
    for i, row in data.iterrows():
        if row.isSetupPickup == 'WAAR':
            row.Annotation = ['PickupSetup']
        elif row.isBlank == 'WAAR':
            row.Annotation = ['Blank']
        elif isinstance(row.animalVernacularName, list):
            row.Annotation = row.animalVernacularName
    
    #Remove row without annotation
    data['Annotation'].replace('', np.nan, inplace=True)
    data.dropna(subset=['Annotation'], inplace=True)
    
    #Adjuct notation of annotation for images with more than one annotation
    for i, row in data.iterrows():
        if len(row.Annotation) > 1:
            row.Annotation = list(set(row.Annotation))
                
    #Initialize lists and output dataframes    
    total_output = None
    boxes_output = None
    
    #Size standard box
    length_standard_box = 960*RATIO
    height_standard_box = 540*RATIO
    
    #Minimum number of pixels object and minimum number of pixels difference
    min_pixel_object = int(6000*RATIO**2)
    min_pixel_diff = int(500*RATIO**2)
    #Maximum number of pixels difference is calculated later, based on the image size.
                    
    #Parameters for binary closing
    struct = np.ones((20,20)).astype(int)
    iter_closing = 5 #number of iterations
    
    #Loop over every deployment
    for folder in os.listdir(resized_folder_path):
        imageFolderPath = os.path.join(resized_folder_path, folder)
        
        #Check if it is a folder, not a file
        if os.path.isdir(imageFolderPath):
            
            annotations_deployment = []
            image_names_sequences = []
    
            data_deployment = data.loc[data['deployment'] == folder]
            sequences = data_deployment.sequenceID.unique()
            
            for seq in sequences:
                image_names_sequences.append(data_deployment.loc[data_deployment['sequenceID'] == seq].filename.tolist())
                annotations_deployment.append(data_deployment[data_deployment['sequenceID'] == seq].Annotation.iloc[0])
    
            lengths = [len(i) for i in image_names_sequences]
            deployment = pd.DataFrame({'ImagesNames': image_names_sequences,'SequenceID': sequences, 'Length':lengths, 'Annotation':annotations_deployment}) #Eventueel nog andere info toevoegen zoals aantal dieren.
            deployment['box_standard'] = ""
            deployment['box_small']= ""
            deployment['deployment']= folder
    
            #Loop over every sequence of the deployment
            for i, row in enumerate(deployment.itertuples(), 1):
                
                # Import images sequence
                images_sequence = pd.DataFrame()
                for img in row.ImagesNames:
                    if os.path.isfile(os.path.join(resized_folder_path,folder,img)):
                        image = Image.open(os.path.join(resized_folder_path, folder, img))
                        name = (image.filename).split('\\')[-1]
                        images_sequence = images_sequence.append(pd.DataFrame([image, name]).T) 
                images_sequence.columns = ['Image', 'ImageName']
                
                if len(images_sequence) == len(row.ImagesNames) and len(images_sequence) > 0: #All images available
                    
                    #Import sequence
                    images_matrices = []
                    series = [] 
                    box_list = []
                    box_list_small = []
                    image_type = []
                    
                    #Import first image to determine the size of the black border for the whole sequence.
                    image_border = images_sequence.iloc[0]['Image']
                    border = black_border(image_border)
                    
                    for rows in images_sequence.itertuples():
                        image = rows.Image
                        image = image.crop(border)
                        
                        #Check if image is a greyscale image
                        if len(set(image.getpixel((length_standard_box,length_standard_box)))) == 1 & len(set(image.getpixel((height_standard_box,height_standard_box)))) == 1: 
                            image = image.convert('L')
                            image_type.append('grey')
                        else:
                            image_type.append('color')
    
                        series.append(image)
                        images_matrices.append(np.asarray(image))
        
                    #Valid sequence? (Remove control images)
                    dim_images = [len(k.shape) for k in images_matrices]
                    if row.Length >= 10 and len(set(dim_images)) == 1:
                        
                        #Calculate the median value of every pixel to determine the background
                        dim = images_matrices[0].ndim
                        image_stack = np.concatenate([im[..., None] for im in images_matrices], axis=dim)
                        median_array = np.median(image_stack, axis=dim)
                        median_image = Image.fromarray(median_array.astype('uint8'))    
                          
                        #Image size
                        image_length = series[0].size[0]
                        image_height = series[0].size[1]
                        
                        #Maximum number of pixels difference
                        max_pixel_diff = image_length*image_height*0.6
                        
                        #Select objects
                        for img in series:
                            
                            #Difference with background
                            diff = ImageChops.difference(median_image, img).convert('L')
                            
                            #MinFilter
                            filter = diff.filter(ImageFilter.MinFilter(size=9))
                            
                            #Number of pixels that are different
                            pixels_filter = cv2.countNonZero(np.asarray(filter))
                            box_filter = filter.getbbox()
                            
                            #No (significant) difference with background
                            if not isinstance(box_filter, tuple) or pixels_filter < min_pixel_diff :
                                length_box_filter = 0
                                height_box_filter = 0
                                box_filter = ()
                                
                                box_list.append(box_filter)
                                box_list_small.append(box_filter)
                            
                            #To much difference with background
                            elif pixels_filter > max_pixel_diff:
                                box_object_list_small = ()
                                box_object_list = devide_box(img.getbbox(), length_standard_box, height_standard_box, image_length, image_height)
                                                      
                                box_list.append(box_object_list)
                                box_list_small.append(box_object_list_small)
                                
                            else: 
                                length_box_filter = size_box(box_filter)[0]
                                height_box_filter = size_box(box_filter)[1]
                            
                                #Box after filtering is smaller than standard box
                                if length_box_filter < length_standard_box and height_box_filter < height_standard_box:
                                    box = standard_box(box_filter, length_standard_box, height_standard_box, image_length, image_height)
                                    box_list.append(box)
                                    box_list_small.append(box_filter)
                                
                                #Box after filtering is larger than standard box
                                else:
                                    #Edge detection
                                    edge = roberts(filter)
                                    
                                    #MinFilter after edge detection
                                    edge = (edge != 0).astype(int)
                                    edge = Image.fromarray(edge.astype('uint8')).filter(ImageFilter.MinFilter(size=3))
                                    edge = Image.fromarray(np.asarray(edge).astype('uint8')).filter(ImageFilter.MinFilter(size=3))
                                    
                                    #Binary closing
                                    closing = ndi.binary_closing(edge, structure=struct, iterations=iter_closing, output=None, origin=0)
                            
                                    #Connected component labeling
                                    connect = measure.label(closing, neighbors=8, background=0, return_num=True)
                                    counts = np.bincount(connect[0].flatten())
                                
                                    #Box after connected component labeling
                                    box = Image.fromarray(closing.astype('uint8')).getbbox()
                                    if not isinstance(box, tuple):
                                        length_box = 0
                                        height_box = 0
                                        box = ()      
                                    else:
                                        length_box = size_box(box)[0]
                                        height_box = size_box(box)[1]
                                
                                    #Box not empty
                                    if length_box != 0:
                                        
                                        #Box after connected component labeling is larger than standard box
                                        if length_box > length_standard_box or height_box > height_standard_box:
                                            
                                            #Boxes around objects
                                            box_object_list = []
                                            box_object_list_small = []
                                            
                                            for a in range(1, (connect[1])+1):
                                                
                                                if counts[a] > min_pixel_object:
                        
                                                    box_object = Image.fromarray((connect[0]==a).astype('uint8')).getbbox()
                                                    box_object_list_small.append(box_object)
                                                    
                                                    length_box_object = size_box(box_object)[0]
                                                    height_box_object = size_box(box_object)[1]
                                                    
                                                    #Box around object bigger than standard box
                                                    if length_box_object > length_standard_box or height_box_object > height_standard_box:
                                                        
                                                        boxes = devide_box(box_object, length_standard_box, height_standard_box, image_length, image_height)
                                                        box_object_list += boxes
                                                        
                                                    #Box around object is smaller than standard box
                                                    else:
                                                        box_object = standard_box(box_object,length_standard_box,height_standard_box, image_length, image_height)
                                                        box_object_list.append(box_object)
                                            
                                            if not box_object_list:
                                                box_object_list = ()
                                                box_object_list_small = ()
                                            box_list.append(box_object_list)
                                            box_list_small.append(box_object_list_small)
                                     
                                        
                                        #Box after connected component labeling is smaller than standard box
                                        else:
                                            box_list_small.append(box)
                                            box = standard_box(box,length_standard_box,height_standard_box, image_length, image_height)
                                            box_list.append(box)
                                    
                                    #Empty box
                                    else: 
                                        box_list.append(box)
                                        box_list_small.append(box)
                                    
                            
                        #Save sequence preprocessing data
                        if all(isinstance(x, (tuple)) for x in box_list):
                            box_list = [[elem] for elem in box_list]
                            df_standard = pd.DataFrame([box_list]).transpose()
                        else:
                            lengths = []
                            for l in range(len(box_list)):
                                item = box_list[l]
                                if isinstance(item, (tuple)):
                                    box_list[l] = [item]       
                                lengths.append(len(box_list[l]))
                                
                            if all(lengths[0] == items for items in lengths):
                                df_standard = pd.DataFrame([box_list]).transpose()
                            else:
                                df_standard = pd.DataFrame(np.array(box_list).reshape(len(row.ImagesNames),-1))
        
        
                        if all(isinstance(x, (tuple)) for x in box_list_small):
                            box_list_small = [[elem] for elem in box_list_small]
                            df_small = pd.DataFrame([box_list_small]).transpose()
                            
                        else:
                            lengths = []
                            for l in range(len(box_list_small)):
                                item_small = box_list_small[l]
                                if isinstance(item_small, (tuple)):
                                    box_list_small[l] = [item_small]        
                                lengths.append(len(box_list_small[l]))
                                
                            if all(lengths[0] == items for items in lengths):
                                df_small = pd.DataFrame([box_list_small]).transpose()
                            else:
                                df_small = pd.DataFrame(np.array(box_list_small).reshape(len(row.ImagesNames),-1))
        
                        boxes_sequence = pd.concat([df_standard,df_small], axis=1)
                        boxes_sequence.columns = ['box_standard', 'box_small']
                        boxes_sequence['deployment'] = folder
                        boxes_sequence['sequence'] = row.SequenceID
                        boxes_sequence['annotation'] = ""
                        for seq_index, seq_row in boxes_sequence.iterrows():
                            seq_row.annotation = row.Annotation
                        boxes_sequence['image_name'] = pd.DataFrame(row.ImagesNames)
                        boxes_sequence['image_type'] = pd.DataFrame(image_type)
                        
                        if boxes_output is None:
                            boxes_output = boxes_sequence
                        else:
                            boxes_output = pd.concat([boxes_output, boxes_sequence], axis = 0)
                        
                        #Save smallest box and standard box
                        deployment.set_value(deployment.index[i-1], 'box_standard', box_list)
                        deployment.set_value(deployment.index[i-1], 'box_small', box_list_small)
                        
            #Save deployment
            if total_output is None:
                total_output = deployment
            else:
                total_output = pd.concat([total_output, deployment])
    
    #Every box => Image
    boxes_single = pd.DataFrame()
    for i, row in boxes_output.iterrows():
        for box in row.box_standard:
            if len(box) != 0:
                boxes_single = boxes_single.append(row, ignore_index=True)
                boxes_single['box_standard'].iloc[-1] = box
            
    #Save data preprocessing        
    total_output.to_csv(os.path.join(preprocessing_output_path, 'data_preprocessing.csv'), index=False, sep=';')
    boxes_output.to_csv(os.path.join(preprocessing_output_path, 'boxes_preprocessing.csv'), index=False, sep=';')
    data.to_csv(os.path.join(preprocessing_output_path, 'data_start_preprocessing.csv'), index=False, sep=';')
    boxes_single.to_csv(os.path.join(preprocessing_output_path, 'boxes_preprocessing_single.csv'), index=False, sep=';')

if __name__ == '__main__':
    preprocessing()