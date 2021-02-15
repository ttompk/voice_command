# Tensorflow Wake Command Identification   

The purpose of this project is to create a wake word on a Raspbery Pi Zero. The task that will be carried out upon wake word identification has not been determined but in the interim an LED light atached to a GPIO pin will flash when the wake word is identified.  

The wake word will be identified from features extracted from real-time audio sampling using a 2-mic reSpeaker breakout board (Seeed Studio) attached to the raspbery pi. Inference of the digital audio input data, i.e. features, will be performed using a tensorflow model trained on a macbook pro which has been convereted to a tensoflow lite model (serialized) and installed on the Pi Zero.    


Approximate Steps:
- Extract features from google's command audio dataset.  
- Train tensorflow model using features for wake word.
- Convert tf model as serialized model using tf-lite
- Load tf-lite model on raspbery pi zero for inference.
- Use speakers to pick up audio near the Raspberry Pi Zero. 
- Evaluate mic audio and infer wake word.
- When wake word has been identified, light up an LED on a breadboard. Eventually will be used to begin recording text and convert the audio to text.
- Example:  Identify the phrase 'set timer for 5 minutes'...begins a countdown for 5 minutes and then sounds an alarm via the speaker attached to the Pi Zero.
  
## Hardware and Environment

My initial plan was to train a model using tensorflow on my mid-2010 MacBook Pro (16GB). I intended to use tensorflow 2.1 or greater as it had keras baked in and Google's reference docs utilized this version in their code. Immediatelly there was an issue loading tensorflow 2.1 or greater as a result of the laptop Intel chip not utilizing AVX. As there was no getting around this requirement I needed to shift development to a Colab notebook...because it's free. I could have easily spun up a machine on Google Cloud Compute...but that's not free and I've never tried developing on Colab.
  

## Training Data

Google Command Dataset: [The docs](https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md)  
[Download directly](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)    

## Feature Extraction

Ref docs for Mel Frequency Cepstral Coefficient: [docs](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)  



 
