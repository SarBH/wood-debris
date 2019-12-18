import cv2
import numpy as np
import json


def convert_to_mask(x_coord, y_coord, sample_name, class_name, ann_img):    

    all_poly_coords = []
    for yx in zip(x_coord, y_coord):
        all_poly_coords.append(yx)
    all_poly_coords = np.array(all_poly_coords)
    
    if class_name == "branch":
        cv2.fillConvexPoly(ann_img, all_poly_coords, 20)
    if class_name == "box":
        cv2.fillConvexPoly(ann_img, all_poly_coords, 60)
    if class_name == "camera-bag":
        cv2.fillConvexPoly(ann_img, all_poly_coords, 100)
    if class_name == "tree":
        cv2.fillConvexPoly(ann_img, all_poly_coords, 140)

    return ann_img
            




if __name__ == "__main__":
    
    IMG_SHAPE = (540,960)

    with open('data/train/via_region_data.json') as mask_json:
        masks = json.load(mask_json)
        for sample_name in masks:
            prev_mask = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1]))
            for region in masks[str(sample_name)]["regions"]:

                class_name = region["region_attributes"]["name"]
                x_coord = region["shape_attributes"]["all_points_x"]
                y_coord = region["shape_attributes"]["all_points_y"]
                
                masked_image = convert_to_mask(x_coord, y_coord, sample_name, class_name, prev_mask)
                prev_mask = masked_image
            
            cv2.imwrite(str("data/train/masks/"+sample_name+".png"), masked_image)