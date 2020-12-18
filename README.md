# wood-debris

This repo contains data and files for the image segmentation task for wood debris detection.

This project acted as a **proof of concept** to the proposal of using machine learning computer vision to quantify the amount of wood debris from photographs. 

The idea is to calculate the area of the 2D image covered by dried out and fallen branches and use it to estimate of the volume of flammable material presented in the image.
Since the photographer is taking these photos from different distances to the objects, an approximate calibration is performed using a set of known objects that were placed in the image near the debris: the "camera-bag" and the "box".

The problem of wood debris quantification on forests was proposed by Prof. Strigul from Stevens Institute of Technology.

## See the Presentation in this repo for proposed solutions and results.

## Contents
### Data:
This folder contains the images and masks used for training and testing. On an instance segmentaiton problem, the labels ("y") are masks of the pixels that make up the objects we are trying to find. In this case, the masks are polygons around the approximate area occupied by branches  and the *control objects*: camera-bag, box.
The masks were created using VGG Image Annotator: https://www.robots.ox.ac.uk/~vgg/software/via/, a free, simple and standalone manual annotation software for images and other types of media.

### Helper Code:
Two scripts: 
- resize-images.py: for resizing images in batches
- convert-to-mask.py: the image annotator outputs a JSON file with the coordinates of the polygon corners of a particular region in an image. This code parses those coordinates
        along with the image's name and the class_name of the region from the JSON and returns the mask for that region as a new image.
        
 
