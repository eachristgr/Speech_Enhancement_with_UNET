"""
Aristotle University of Thessaloniki

Course : Audio & Video Technology (2020 - 2021)
Project : Speech Enhancement using U-Net
Authors : Emmanouil Christos, Amoiridis Vasilios, Anagnostou Athanasios, Tsoukias Stefanos

Sprit : 05 - Create Enchanced Spects for Testing 
"""

##################################################
# Imports
##################################################
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import backend
print("Tensorflow version: " + tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


##################################################
# Main
##################################################
#-------------------------------------------------
# Prepare Input Data
#-------------------------------------------------
# Load Test Data
x_spect = np.load('Dataset_My_Spects\\Testing_NoisySpeech_spect_arrary.npy')


# Normalize Data
minX_array = np.zeros((x_spect.shape[0], 1), np.float32)
maxX_array = np.zeros((x_spect.shape[0], 1), np.float32)
for i in range(x_spect.shape[0]):
    minX_array[i] = x_spect[i,:,:].min()
    x_spect[i,:,:] = x_spect[i,:,:] - minX_array[i]
    maxX_array[i] = x_spect[i,:,:].max()
    x_spect[i,:,:] = x_spect[i,:,:] / maxX_array[i]
     
    
# Resize
x_spect = x_spect.reshape(x_spect.shape[0], x_spect.shape[1], x_spect.shape[2], 1)


#-------------------------------------------------
# Load Model
#-------------------------------------------------
json_file = open('UNET_Model_Saves\\UNET_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
my_unet = model_from_json(loaded_model_json)
# load weights int new model
my_unet.load_weights('UNET_Model_Saves\\UNET_Model_Weights.h5')
print("Loaded model from disk")

#-------------------------------------------------  
# Get Output Data
#-------------------------------------------------  
predicted_spect = my_unet.predict(x_spect)
y_spect = predicted_spect # NoisySpeech -> CleanSpeech
#y_spect = np.subtract(x_spect, predicted_spect) # NoisySpeech -> Noise


# Denormalize
for i in range(y_spect.shape[0]):
    y_spect[i,:,:] = (y_spect[i,:,:] * maxX_array[i]) + minX_array[i]
    
    
# Save Spects
np.save('Dataset_My_Spects\\Testing_EnhancedSpeech_spect_arrary', np.array(y_spect[:,:,:,0])) 