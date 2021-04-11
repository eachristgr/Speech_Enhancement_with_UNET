# Speech Enhancement with UNET

##### **Aristotle University of Thessaloniki - Electrical and Computer Engineering**

##### **Course : Audio & Video Technology**

##### **Authors : [Emmanouil Christos](https://github.com/eachristgr), [ Amoiridis Vasilios](https://github.com/vamoirid), [Anagnostou Athanasios](https://github.com/Nassos-Anagnostou), [Tsoukias Stefanos](https://github.com/tsoukias)**

------

This repository contains the assignment for the course of Audio & Video Technology.

The aim was to get to know the branch of deep learning and to apply it to the problem of the de-noiseization of human speech.

## Dataset

The dataset used is [MS-SNSD](https://github.com/microsoft/MS-SNSD) (Microsoft Scalable Noisy Speech Database).

With the help of the functions it provides and after selecting specific types of noises, they were mixed with the clear speech signals in various SNR ratios (0 dB, 5 dB, 10 dB, 15 dB, 20 dB) and thus a total of 4 hours training set and a 30min test set were created. This procedure can be found in the **s01_CreateWAVs.py** file.

In the **Dataset_MS_SNSD** and **Dataset_My_Wavs** folders there are screenshots that show how the audio files were placed in the original and final sets.

## UNET

The model used can be found in the **s03_InitializeModel.py** file and can be seen in the following image:

<img src="https://github.com/eachristgr/Speech_Enhancement_with_UNET/blob/master/UNET_Model_Saves/UNET_Shape.png?raw=true"/> 

Note that the input of the model is noisy speech magnitude spectrometers while as an output two cases were examined. In the first the output was the magnitude spectrograms of the noise and in the second the magnitude spectrograms of the clean speech.

Τhe **s00_tools.py** and **s02_CreateSpects.py** files show the process by which the spectrograms were created and in the **Dataset_My_Spects** folder there is an screenshot that show how the spectrograms were saved.

Also in **s04_TrainModel.py** and **s05_CreateEnhancedSpectsForTest.py** files it can bee seen how the model was trained and how the enhanced magnitude spectrograms were created. The final trained model is save in the **UNET_Model_Saves** folder.

## Evaluation

In order to evaluate the model the [Segmental SNR, Frequency-Weighted Segmental SNR](https://github.com/schmiph2/pysepm) and [Short-Time Objective Intelligibility (STOI)](https://github.com/mpariente/pystoi) metrics were used.

In the **s06_CreateEnhancedAudioAndEvaluate.py** file can be seen how the final audios ere created using the enhanced magnitude and initial phase spectrograms as well as how the evaluation was made.

## How to Use

In order to use the final trained model in order to de-noise any audio file, a routine has been created which is shown in the **s07_UseUNetExample.py** file.

If you want to test the model in your audio file you have to change the filename_NoisySpeech variable in the line 39. Some audio samples can be found in the **s07_AudioExamples** folder.

------

- It is noted that the complete research that was done, more details of the procedure that was followed but also the test results can be found in the **report.pdf** which is written in Greek.
