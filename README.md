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
  

### Seeed Studios 2-mic respeaker
hat wiki:  https://wiki.seeedstudio.com/ReSpeaker_2_Mics_Pi_HAT/   
Since a raspberry pi zero lacks audio input and output capabilities (aside from HDMI), I needed to find a method to get our wake words to our model for inference. The 2-mic board is capable for both audio input and audio output. The manufacturer describes it as:  
 
_The 2-mic respeaker hat contains 2 mReSpeaker 2-Mics Pi HAT is a dual-microphone expansion board for Raspberry Pi designed for AI and voice applications._   

The board fits GPIO pins on:  
- raspberrry pi board (including zero)  
- NPi i.MX6ULL Dev Board Linux SBC  
- ODYSSEY - STM32MP157C
- Nvidia Jetson Nano Series

### Installation of 2-mic respeaker on Raspberry Pi Zero W
These instructions assume you are running RaspberryOS Buster  

__Step 1__  
1. Update/upgrade the Pi Zero.  
`sudo apt-get update`  
`sudo apt-get upgrade`
2. Clone the seeed-voicecard repo:   
`git clone https://github.com/respeaker/seeed-voicecard.git`   
3. Run the _install.sh_ bash script in the new repo  
`sudo seeed-voicecard/install.sh`  
4. Reboot  
`sudo reboot`   

__Step 2__  
1. Verify the card has been installed and the playback device is detected by the Pi. Verify the output of this command matches the [wiki](https://wiki.seeedstudio.com/ReSpeaker_2_Mics_Pi_HAT/).   
`aplay -l`  
2. Similarly, verify the audio input device is also listed (same wiki).   
`arecord -l`  
3. Test playback on the speaker. Can use headphones or plug a speaker to into the 2-pin JST PH (2.0mm pitch) connector.  I've written two methods to play the sounds back, the first requires headphones. This method uses the microphones on the card to record a few seconds of sound and then plays that sound back on the headphones, in a record/play loop. If you run this comand with a speaker attached you will encounter a horribly loud shreeking sound due to the feedback between speaker and mics! The 'speaker' method requires testing the speakers separately, first play back of a wav file in the home directory followed by a separate mic input.   
Warning!! Do not use the following command with speakers or your eardrums will burst!!  
Headphones Only (_Ctrl+C to exit):  
`arecord -f cd -Dhw:1 | aplay -Dhw:1`   
With speaker:   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp - Download _piano2.wav_ from this repo and copy to the Pi Zero. I use scp to drop in the Pi's home folder,  e.g.: 
`scp piano2.wav pi@xxx.xxx.x.xxx:~`   
    * Play the tune using the speaker attached to seeed-studio card (_Ctrl+C_ to exit):   
`aplay -Dhw:1 -d 10 piano2.wav`   
[Reference on playing tunes from command line in linux.](https://www.systutorials.com/docs/linux/man/1-speaker-test/)  


## Training Data

Google Command Dataset: [The docs](https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md)  
[Download directly](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)    

## Feature Extraction

Ref docs for Mel Frequency Cepstral Coefficient: [docs](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)  



 
