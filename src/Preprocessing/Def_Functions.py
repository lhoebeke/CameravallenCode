import math
import numpy as np
   
def standard_box(box, standard_length, standard_height, image_length, image_height):
    
   """Returns the boundaries of a box with specified length en height, centered around a given smaller box. 
   The obtained bigger box lies within the image.
   
   Arguments:
   box: original smaller box
   standard_length: length of the standard box
   standard_heighth: height of the standard box
   image_length: length of the image
   image_height: height of the image
   
   Return:
   box_standard: boundaries of the standard box (left, top, right, bottom)
   
   """

   
   box = np.asarray(box)
   length_box = box[2]-box[0]
   height_box = box[3]-box[1]
   
    
   if box[0]-(standard_length-length_box)/2 < 0: #left outside the image
        box[0] = 0
        box[2] = standard_length
   elif box[2]+(standard_length-length_box)/2 > image_length: #right outside the image
        box[0] = image_length-standard_length-1
        box[2] = image_length-1
   else:
        box[0] = box[0]-math.floor((standard_length-length_box)/2)
        box[2] = box[2]+math.ceil((standard_length-length_box)/2)
    
    
   if box[1]-(standard_height-height_box)/2 < 0: #top outside the image
        box[1] = 0
        box[3] = standard_height
   elif box[3]+(standard_height-height_box)/2 > image_height: #bottom outside the image
        box[1] = image_height-standard_height-1
        box[3] = image_height-1
   else:
        box[1] = box[1]-math.floor((standard_height-height_box)/2)
        box[3] = box[3]+math.ceil((standard_height-height_box)/2)
        
   box_standard = tuple(box)

   return box_standard

##############################################################################
def black_border(image):
    
   """Returns the boundaries of the image without the black borders at the top and bottom of the image, containing metadata.
   
   Arguments:
   image: the image

   Return:
   image_box: boundaries of the image without the borders (left, top, right, bottom)
   
   """

   image = image.convert('L')
   image_length = image.size[0]
   image_height = image.size[1]
   
   column = np.asarray(image)[:,0]
   top = next((i for i, x in enumerate(column) if x))
   bottom = (image_height-1) - next((i for i, x in enumerate(column[::-1]) if x))
   image_box = tuple([0, top, image_length-1, bottom])
   
   return image_box

##############################################################################
def size_box(box):
    
   """Calculates the length and the height of a box.
   
   Arguments:
   box: boundaries of the box (left, top, right, bottom)

   Return:
   length_box: length of the box
   height_box: height of the box
   
   """
   length_box = box[2]-box[0]
   height_box = box[3]-box[1]
   
   return [length_box, height_box]

##############################################################################
def devide_box(box, length_standard_box, height_standard_box, image_length, image_height):
    
   """Devides a box that is bigger than the standard box, into standard boxes.
   The obtained standard boxes lie within the image.
   
   Arguments:
   box: original box
   standard_length: length of the standard box
   standard_heighth: height of the standard box
   image_length: length of the image
   image_height: height of the image
   
   Return:
   boxes: list containing the boundaries of the boxes (left, top, right, bottom)
   """
   length_box = size_box(box)[0]
   height_box = size_box(box)[1]
   
   #Option 1: box is longer than the standard box, but not higher                       
   if length_box > length_standard_box and height_box < height_standard_box:
    
       left_box = np.asarray(box)
       left_box[2] = left_box[0]+length_standard_box
       left_box = standard_box(left_box, length_standard_box, height_standard_box, image_length, image_height)
        
       right_box = np.asarray(box)
       right_box[0] = right_box[2]-length_standard_box
       right_box = standard_box(right_box,length_standard_box,height_standard_box, image_length, image_height)
       
       boxes = [tuple(left_box), tuple(right_box)]
       
   #Option 2: box is higher than the standard box, but not longer                       
   elif length_box < length_standard_box and height_box > height_standard_box:
    
       top_box = np.asarray(box)
       top_box[3] = top_box[1]+height_standard_box
       top_box = standard_box(top_box, length_standard_box, height_standard_box, image_length, image_height)
        
       bottom_box = np.asarray(box)
       bottom_box[1] = bottom_box[3]-height_standard_box
       bottom_box = standard_box(bottom_box,length_standard_box,height_standard_box, image_length, image_height)
       
       boxes = [tuple(top_box), tuple(bottom_box)]

   #Option 3: box is higher and longer than the standard box                     
   else:
    
       topleft_box = np.asarray(box)
       topleft_box[2] = topleft_box[0]+length_standard_box
       topleft_box[3] = topleft_box[1]+height_standard_box

       topright_box = np.asarray(box)
       topright_box[0] = topright_box[2]-length_standard_box
       topright_box[3] = topright_box[1]+height_standard_box
       
       bottomleft_box = np.asarray(box)
       bottomleft_box[2] = bottomleft_box[0]+length_standard_box
       bottomleft_box[1] = bottomleft_box[3]-height_standard_box
       
       bottomright_box = np.asarray(box)
       bottomright_box[0] = bottomright_box[2]-length_standard_box
       bottomright_box[1] = bottomright_box[3]-height_standard_box
      
       boxes = [tuple(topleft_box), tuple(topright_box), tuple(bottomleft_box), tuple(bottomright_box)]
       
   return boxes

###############################################################################
def remove_dup_columns(frame):
        
   """
   Removes duplicate columns (same column name) from dataframe.
   """
   
   keep_names = set()
   keep_icols = list()
   for icol, name in enumerate(frame.columns):
       if name not in keep_names:
              keep_names.add(name)
              keep_icols.append(icol)
   return frame.iloc[:, keep_icols]

################################################################################
def box_center(box):
    
   """Returns the coördinates of the center of a box.
   
   Arguments:
   box: box (left, top, right, bottom)

   Return:
   center: (left, top)
   
   """
   box = np.array(box)
   left = math.ceil(np.mean([box[0], box[2]]))
   top = math.ceil(np.mean([box[1], box[3]]))
   center = tuple((left,top))
   
   return center
#################################################################################