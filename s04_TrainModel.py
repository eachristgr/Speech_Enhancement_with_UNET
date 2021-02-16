"""
Aristotle University of Thessaloniki

Course : Audio & Video Technology (2020 - 2021)
Project : Speech Enhancement using U-Net
Authors : Emmanouil Christos, Amoiridis Vasilios, Anagnostou Athanasios, Tsoukias Stefanos

Sprit : 04 - Train U-Net & Save model
"""

##################################################
# Imports
##################################################
import s03_InitializeModel as unet
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import callbacks
print("Tensorflow version: " + tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


##################################################
# Main
##################################################
# Load Train X, Y (NoisySpeech -> CleanSpeech)
x_train = np.load('Dataset_My_Spects\\Training_NoisySpeech_spect_arrary.npy')
y_train = np.load('Dataset_My_Spects\\Training_CleanSpeech_spect_arrary.npy')


# Normalize Data -> pws kanonikopoiw to Y?
for i in range(x_train.shape[0]):
    min_x = x_train[i,:,:].min()
    x_train[i,:,:] = x_train[i,:,:] - min_x
    maxX = x_train[i,:,:].max()
    x_train[i,:,:] = x_train[i,:,:] / maxX
    
    y_train[i,:,:] = (y_train[i,:,:] - min_x) / maxX
    """
    min_y = y_train[i,:,:].min()
    y_train[i,:,:] = y_train[i,:,:] - min_y
    maxy = y_train[i,:,:].max()
    y_train[i,:,:] = y_train[i,:,:] / maxy
    """


# Reshape to fit input layer
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)


# Inilitalize Model
my_unet = unet.my_unet()
#my_unet.summary()


#-------------------------------------------------
# Train U-Net Model
#-------------------------------------------------
# Ti loss function thelw ?????
# tf.keras.losses.Huber()
# tf.keras.losses.mean_squared_error
# tf.keras.losses.mean_absolute_error
my_unet.compile(optimizer = tf.keras.optimizers.Adam(lr=5e-4), loss = tf.keras.losses.mean_squared_error, metrics = [])
callback = [callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
history = my_unet.fit(x_train, y_train, batch_size=5, shuffle=True, epochs=20, validation_split=0.2, callbacks=callback, verbose=2)


# Plot training & validation accuracy values
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.yscale('log')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#-------------------------------------------------
# Save Model
#-------------------------------------------------
saveDIR = 'UNET_Model_Saves'

model_json = my_unet.to_json()
with open(saveDIR + '\\' + 'UNET_Model.json', 'w') as json_file:
    json_file.write(model_json)
my_unet.save_weights(saveDIR + '\\' + 'UNET_Model_Weights.h5')
print('Saved model to disk')
my_unet.save(saveDIR + '\\' + 'Saved_Model')