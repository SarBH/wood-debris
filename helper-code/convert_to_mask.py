import cv2
import numpy as np
import json


def convert_to_mask(x_coord, y_coord, sample_name, class_name, ann_img):
    """Given the coordinates of the polygon corners of a particular region in an image,
        along with the image's name and the class_name of the region, returns the mask for that region.
        Intended to be called iteratively over all regions in the image. See main method"""
    
    # coordinates must be joined to x-y pairs and into a numpy array 
    # because thats how cv2's fillConvexPoly() wants them
    all_poly_coords = []
    for xy in zip(x_coord, y_coord):
        all_poly_coords.append(xy)
    all_poly_coords = np.array(all_poly_coords)
    
    # depending of the class of that region, paint it a different shade
    if class_name == "branch":
        cv2.fillConvexPoly(ann_img, all_poly_coords, 2)
    if class_name == "box":
        cv2.fillConvexPoly(ann_img, all_poly_coords, 3)
    if class_name == "camera-bag":
        cv2.fillConvexPoly(ann_img, all_poly_coords, 4)
    if class_name == "tree":
        cv2.fillConvexPoly(ann_img, all_poly_coords, 5)

    # return mask with added region
    return ann_img
            




if __name__ == "__main__":
    
    # define the dimensions of the original images, masks must keep the same dimensions
    IMG_SHAPE = (540,960)

    # open json file that came from via annotations. Deselect all three options when downloading
    with open('data/train/via_region_data.json') as mask_json:
        json_file = json.load(mask_json)
        
        # iterate through all "image" objects to extract data
        for image in json_file["_via_img_metadata"].values():
            filename = image["filename"]
            
            prev_mask = np.ones((IMG_SHAPE[0], IMG_SHAPE[1]))
            
            # fail-safe: if no regions were defined for an image, skip it
            if image["regions"] == []:
                continue

            # iterate through all regions to create masks
            for region in image["regions"]:

                class_name = region["region_attributes"]["name"]
                x_coord = region["shape_attributes"]["all_points_x"]
                y_coord = region["shape_attributes"]["all_points_y"]
                
                # call convert_to_mask to make polygon coordinates -> masks in image
                masked_image = convert_to_mask(x_coord, y_coord, filename, class_name, prev_mask)
                prev_mask = masked_image
            
            # when done with all regions defined for an image, save the mask.
            image_tensor = np.dstack([masked_image, masked_image, masked_image])
            cv2.imwrite(str("data/train/masks/"+filename+".png"), image_tensor)