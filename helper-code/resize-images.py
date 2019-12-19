import os
import cv2

orig_path = '/home/sarita/Documents/wood-debris/data/train/masks' # path to the original full sized images
resized_path = '/home/sarita/Documents/wood-debris/resized/masks' # path to place resized images

# iterate through all images in path
for filename in os.listdir(orig_path):
    img = cv2.imread(os.path.join(orig_path, filename))
    print('Orig. dimensions', img.shape)

    # # resize images by scale_percent %
    # scale_percent = 25
    # width = int(img.shape[1] * scale_percent/100)
    # height = int(img.shape[0] * scale_percent/100)
    
    width = 960
    height = 544
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized dimensions', resized.shape)

    # enter name for resized image: I keep the same name with a _r at the end
    new_img_name = filename[:-4] + '_r.JPG'

    # export the resized image
    cv2.imwrite(os.path.join(resized_path, new_img_name), resized)
