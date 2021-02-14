# Tensorflow Wake Command Identification   

The purpose of this project is to create a wake word on a Raspbery Pi Zero. The task that will be carried out upon wake word identification has not been determined but in the interim an LED light atached to a GPIO pin will flash when the wake word is identified.  

The wake word will be identified from features extracted from real-time audio sampling using a 2-mic reSpeaker breakout board (Seeed Studio) attached to the raspbery pi. Inference of the digital audio input data, i.e. features, will be performed using a tensorflow model trained on a macbook pro which has been convereted to a tensoflow lite model (serialized) and installed on the Pi Zero.    


- Extract features from google's command audio dataset: File is  
- Train tensorflow model using features for wake word.
- Convert tf model as serialized model using tf-lite
- Load tf-lite model on raspbery pi zero for inference.
- After responding to wake word, begin recording text and convert to text to.
- identify 'set timer for X minutes'


 
