"""
Aristotle University of Thessaloniki

Course : Audio & Video Technology (2020 - 2021)
Project : Speech Enhancement using U-Net
Authors : Emmanouil Christos, Amoiridis Vasilios, Anagnostou Athanasios, Tsoukias Stefanos

Sprit : 07 - Use trained UNET model to enhance an wav file of every size.
"""

##################################################
# Imports
##################################################
import numpy as np
import librosa
import s00_tools as tls
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import backend
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import write
print("Tensorflow version: " + tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


##################################################
# Main
##################################################
#-------------------------------------------------
# Set Parameters
#-------------------------------------------------
filename_NoisySpeech = 's07_AudioExamples\\NoisySpeech01.wav'
sampling_rate = 16000


#-------------------------------------------------
# Load & Prepare Audio
#-------------------------------------------------
# Load wav file
wav_NoisySpeech, fs = librosa.load(filename_NoisySpeech, sampling_rate)


# Make sure the audio has a length tha can parse it to frames of same length.
# We want frames to have frame_sec seconds of audio.
# If the audio has no the righ size we add a zero pad at the end.
num_of_samples_Original = wav_NoisySpeech.shape[0]
num_of_samples_PerFrame = 32512

zeros_size = num_of_samples_Original + (num_of_samples_PerFrame - (num_of_samples_Original % num_of_samples_PerFrame))
wav_NoisySpeech_CorrectSize = np.zeros((zeros_size, ), np.float32)
wav_NoisySpeech_CorrectSize[0:num_of_samples_Original] = wav_NoisySpeech[0:num_of_samples_Original]


# Parse Audio into Frames
num_of_frames = wav_NoisySpeech_CorrectSize.shape[0] // num_of_samples_PerFrame
wav_NoisySpeech_Frames = np.zeros((num_of_samples_PerFrame, num_of_frames), np.float32)

for i in range(num_of_frames):
    wav_NoisySpeech_Frames[:,i] = wav_NoisySpeech_CorrectSize[i*num_of_samples_PerFrame:i*num_of_samples_PerFrame+num_of_samples_PerFrame]
    

# Compute Frames Magnitude & Phase Spectograms
mag_spect_Frames = np.zeros((num_of_frames, 256, 256), np.float32)
phase_spect_Frames = np.zeros((num_of_frames, 256, 256), np.float32)

for i in range(num_of_frames):
    mag_spect_Frames[i,:,:], phase_spect_Frames[i,:,:] = tls.audio_to_spect(wav_NoisySpeech_Frames[:,i])
    

# Normalize Every Spectogram of Every Frame
minX_array = np.zeros((num_of_frames, 1), np.float32)
maxX_array = np.zeros((num_of_frames, 1), np.float32)
for i in range(num_of_frames):
    minX_array[i] = mag_spect_Frames[i,:,:].min()
    mag_spect_Frames[i,:,:] = mag_spect_Frames[i,:,:] - minX_array[i]
    maxX_array[i] = mag_spect_Frames[i,:,:].max()
    mag_spect_Frames[i,:,:] = mag_spect_Frames[i,:,:] / maxX_array[i]
    

#-------------------------------------------------
# Load & Use Model
#-------------------------------------------------
# Load Model
json_file = open('UNET_Model_Saves\\UNET_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
my_unet = model_from_json(loaded_model_json)
# load weights int new model
my_unet.load_weights('UNET_Model_Saves\\UNET_Model_Weights.h5')


# Make predictions for every frame
mag_spect_Frames_Predicted = my_unet.predict(mag_spect_Frames)
mag_spect_Frames_Predicted = mag_spect_Frames_Predicted[:,:,:,0]


#-------------------------------------------------
# Create Enhanced Audio
#-------------------------------------------------
# Denormalize Data
for i in range(num_of_frames):
    mag_spect_Frames_Predicted[i,:,:] = (mag_spect_Frames_Predicted[i,:,:] * maxX_array[i]) + minX_array[i]
    

# Create Audio For Every Frames
wav_EnhancedSpeech_Frames = np.zeros((32385, num_of_frames), np.float32)
for i in range(num_of_frames):
    wav_EnhancedSpeech_Frames[:, i] = tls.spect_to_audio(mag_spect_Frames_Predicted[i,:,:], phase_spect_Frames[i,:,:])
    

# Combine Every Frame
wav_EnhancedSpeech = np.zeros((32385 * num_of_frames), np.float32)
for i in range(num_of_frames):
    wav_EnhancedSpeech[i*32385:i*32385+32385] = wav_EnhancedSpeech_Frames[:, i]

#wav_EnhancedSpeech = wav_EnhancedSpeech[0:num_of_samples_Original]


#-------------------------------------------------
# Play Audio & Plot Waveforms, Spectograms
#-------------------------------------------------
# NoisySpeech
sd.play(wav_NoisySpeech, 16000)
status = sd.wait()

scipy.io.wavfile.write('Waveform_Noisy_Speech', 16000, wav_NoisySpeech)
plt.plot(wav_NoisySpeech)
plt.title('Waveform_Noisy_Speech')
plt.show()

mag_spect_NoisySpeech, phase_spect_NoisySpeech = tls.audio_to_spect(wav_NoisySpeech)
plt.figure(figsize=(10,10))
plt.imshow(mag_spect_NoisySpeech, origin='lower')
plt.title('Magnitude_Spectogram_Noisy_Speech')
plt.show()


# EnhancedSpeech
sd.play(wav_EnhancedSpeech, 16000)
status = sd.wait()

scipy.io.wavfile.write('Waveform_Enhanced_Speech', 16000, wav_EnhancedSpeech)
plt.plot(wav_EnhancedSpeech)
plt.title('Waveform_Enhanced_Speech')
plt.show()

mag_spect_EnhancedSpeech, phase_spect_EnhancedSpeech = tls.audio_to_spect(wav_EnhancedSpeech)
plt.figure(figsize=(10,10))
plt.imshow(mag_spect_EnhancedSpeech, origin='lower')
plt.title('Magnitude_Spectogram_Enhanced_Speech')
plt.show()
