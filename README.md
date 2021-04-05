# Speech Enhancement with UNET

##### **Aristotle University of Thessaloniki - Electrical and Computer Engineering**

##### **Course : Video & Audio Technology**

##### **Authors : [Emmanouil Christos](https://github.com/eachristgr), [ Amoiridis Vasilios](https://github.com/vamoirid), [Anagnostou Athanasios](https://github.com/Nassos-Anagnostou), [Tsoukias Stefanos](https://github.com/tsoukias)**

------

This repository contains the assignment for the course of Video & Audio Technology.

The aim was to get to know the branch of deep learning and to apply it to the problem of the de-noiseization of human speech.

## Dataset

The dataset used is [MS-SNSD](https://github.com/microsoft/MS-SNSD) (Microsoft Scalable Noisy Speech Database).

With the help of the functions it provides and after selecting specific types of noises, they were mixed with the clear speech signals in various SNR ratios (0 dB, 5 dB, 10 dB, 15 dB, 20 dB) and thus a total of 4 hours training set and a 30min test set were created. This procedure can be found in the **s01_CreateWAVs.py** file.

In the Dataset_MS_SNSD and Dataset_My_Wavs folders there are screenshots that show how the audio files were placed in the original and final sets.

## UNET

The model used can be found in the **s03_InitializeModel.py** file and can be seen in the following image:

<img src="https://github.com/eachristgr/Speech_Enhancement_with_UNET/blob/master/UNET_Model_Saves/UNET_Shape.png?raw=true" width="200"/> 

