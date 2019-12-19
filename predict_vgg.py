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

from image_segmentation_keras_local.keras_segmentation_local.predict import predict

predict( 
	checkpoints_path="model_data/v2_model_data/vgg_wood-debris_v2", 
	inp="data/val/images/DSC01671_r.jpg", 
	out_fname="model_data/v2_model_data/predictions/DSC01671_r.jpg" 
)