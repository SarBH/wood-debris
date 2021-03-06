# from image_segmentation_keras_local import keras_segmentation_local

from image_segmentation_keras_local.keras_segmentation_local.models.unet import vgg_unet
from image_segmentation_keras_local.keras_segmentation_local import __init__

__init__


# MUST ALWAYS PASTE THIS ON EVERY SCRIPT BECAUSE RTX CARDS REQUIRE IT
#   otherwise will see this error: 
        #   UnknownError:  Failed to get convolution algorithm. 
        #   This is probably because cuDNN failed to initialize, 
        #   so try looking to see if a warning log message was printed above.
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = vgg_unet(n_classes=5 ,  input_height=544, input_width=960)

model.train(
    train_images =  "data/train/images/",
    train_annotations = "data/train/masks/",
    checkpoints_path = "tmp/vgg_wood-debris_v2" , epochs=5
)

out = model.predict_segmentation(
    inp="data/val/images/DSC01670_r.jpg",
    out_fname="tmp/predictions/DSC01670_r.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)

# # evaluating the model 
# print(model.evaluate_segmentation( inp_images_dir="data/val/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )