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

My initial plan was to train a model using tensorflow on my mid-2010 MacBook Pro (16GB). I intended to use tensorflow 2.1 or greater as it had keras baked in and Google's reference docs utilized this version in their code. Immediately there was an issue loading tensorflow 2.1 or greater as a result of the laptop Intel chip not utilizing AVX. As there was no getting around this requirement I needed to shift development to a Colab notebook...because it's free. I could have spun up a machine on Google Cloud Compute...but that's not free and since I've never tried developing on Colab I thought I'd give it a shot.

__Development Computer__
- I specifically used a mid-2010 MacBook Pro...but feel free to use whatever tool you want as long as you have tensorflow 2.1 or greater installed.  

__Inference Machine__
- Raspberry Pi Zero W. One would presume a non-zero raspberry pi would work great...but I purposely choose a machine that was more resource constrained to see how well it performed. The goal of the project is performing lightweight inference.   
- 2.5A 5V micro-USB power cable for Pi Zero.
- Seeed Studios 2-mic reSpeaker breakout board  
- 8ohm ~2" speaker  
- Female JST PH connector (2pin, 2.0mm pitch) with wires    

## Software
- Rasberrypi OS (Buster)
- Tensorflow 2.1+  (less than 2.1 does not have tensorflow lite)
- reSpeaker install code: git clone https://github.com/respeaker/seeed-voicecard.git
- ALSA  (sound control - built-in to RPi OS  
- RPi.GPIO  (interact with GPIO pins)  


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
3. Test playback on the speaker. Can use headphones or plug a speaker to into the 2-pin JST PH (2.0mm pitch) connector.  I've written two methods to play the sounds back, the first requires headphones. This method uses the microphones on the card to record a few seconds of sound and then plays that sound back on the headphones, in a record/play loop. If you run this comand with a speaker attached you will encounter a horribly loud shreeking sound due to the feedback between speaker and mics!  The 'speaker' method requires testing the speaker and mic separately. First play back of a wav file in the home directory followed by a separate mic recording and playback.   
Warning!! Do not use the following command with speakers or your eardrums will burst!!  
- __Headphones Only__. You will need to replace the '1' in '-Dhw:1' with the number of the card from 'aplay -l' (_Ctrl+C to exit):  
`arecord -f cd -Dhw:1 | aplay -Dhw:1`   
- With speaker or headphones. You will need to replace the '1' in '-Dhw:1' with the correct output device number:     
    * Record 4 seconds of sound through the mics then playback through the speakers, not simulatanouesly.    
`arecord -f S16_LE -d 5 -r 16000 -Dhw:1 /tmp/test-mic.wav -c 2 &&  aplay -Dhw:1 /tmp/test-mic.wav`  
[Reference on playing tunes from command line in linux.](https://www.systutorials.com/docs/linux/man/1-speaker-test/)  

- To test playback only: 
    * Download _piano2.wav_ from this repo and copy to the Pi Zero. I use scp to drop in $
`scp piano2.wav pi@xxx.xxx.x.xxx:~`
    * Play the tune using the speaker attached to seeed-studio card (_Ctrl+C_ to exit):
`aplay -Dhw:1 -d 10 piano2.wav`

If the playback is faint or too loud, don't worry. We can adjust it using the built-in alsa mixer.  

4. Run the AlsaMixer from the command line:   
`alsamixer`  
Press F6 to select the 'seeed-voicecard' sound card.  There are a ton of options the move around...but I'm not sure what they do...perhaps I'll look into it later to fine tune the input/output. 

5. There is a driver for the chip for the included LED but I'm leaving them alone for now.  

6. User Button:  the on-board user button is connected to GPIO17.  
see sample code here to test button:  [ref](https://wiki.seeedstudio.com/ReSpeaker_2_Mics_Pi_HAT/)  



## Training Data

Google Command Dataset: [The docs](https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md)  
[Download directly](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)    

## Feature Extraction

Ref docs for Mel Frequency Cepstral Coefficient: [docs](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)  



 
# Tensorflow Lite 
## Adding TensorFlow Lite to Pi Zero
To add tensorflow lite to the Pi Zero, it must be compiled natively on the Zero using CMake. I tried the instructions [here](https://www.tensorflow.org/lite/guide/build_rpi#compile_natively_on_raspberry_pi) without success.  There are two methods to install TF Lite. Both are provided by separate TF lite documents but the first method did not work on my Pi Zero but I was hopeful it would.   
  
  * __Perform this for both methods__   
From Google: [ref](https://www.tensorflow.org/lite/guide/build_rpi#compile_natively_on_raspberry_pi)  
`sudo apt-get install build-essential`  
`git clone https://github.com/tensorflow/tensorflow.git tensorflow_src`    
`cd tensorflow_src && ./tensorflow/lite/tools/make/download_dependencies.sh`  

  * __Method 1:__   
compile using build_rpi_lib.sh  
Try this version and if does not work then try version 2.  
(The following statement does not work.)   
`./tensorflow/lite/tools/make/build_rpi_lib.sh`  
 
The above statement does not work - the installer is referencing '-DTFLITE_WITHOUT_XNNPACK -march=armv7-a' and similar statements like '/home/pi/tensorflow_src/tensorflow/lite/tools/make/gen/rpi_armv7l' which appears to be arm v7, whereas the Pi Zero is arm v6.  
  
  * __Method 2:__   
From Google: [ref](https://www.tensorflow.org/lite/guide/build_cmake_arm)
`curl -L https://github.com/rvagg/rpi-newer-crosstools/archive/eb68350c5c8ec1663b7fe52c742ac4271e3217c5.tar.gz -o rpi-toolchain.tar.gz
`tar xzf rpi-toolchain.tar.gz -C ${HOME}/toolchains`  
`mv ${HOME}/toolchains/rpi-newer-crosstools-eb68350c5c8ec1663b7fe52c742ac4271e3217c5 ${HOME}/toolchains/arm-rpi-linux-gnueabihf

`ARMCC_PREFIX=${HOME}/toolchains/arm-rpi-linux-gnueabihf/x64-gcc-6.5.0/arm-rpi-linux-gnueabihf/bin/arm-rpi-linux-gnueabihf-`   
`ARMCC_FLAGS="-march=armv6 -mfpu=vfp -funsafe-math-optimizations"`   
(The following does not work. Requires CMake 3.16)  
`cmake -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \`  
`  -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \`  
`  -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" \`   
`  -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \`  
`  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \`  
`  -DCMAKE_SYSTEM_NAME=Linux \`  
`  -DCMAKE_SYSTEM_PROCESSOR=armv6 \`  
`  -DTFLITE_ENABLE_XNNPACK=OFF \`   
`  ../tensorflow/lite/`  

Method 2 will not work because the code requires CMake version 3.16 whereas the 'proper' CMake version for RPi OS Buster is 3.13. 

## Installing CMake v3.16.1
Raspberry Pi OS Buster comes with 3.13 but apt-get upgrade will not update to 3.16. In order to upgrade so you must do this:   

1. CMake requires OpenSSL and cannot find it unless you run the following:  
`sudo apt-get install libssl-dev`  

2. Run the following commands to install CMake:  
  * Step 1  
`version=3.16`  
`build=1`  
`mkdir ~/temp`  
`cd ~/temp`  
`wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz`
`tar -xzvf cmake-$version.$build.tar.gz`
`cd cmake-$version.$build/`

  * Step 2  
`./bootstrap`  
`make -j$(nproc)`  
`sudo make install`  
`cmake --version`  

/home/pi/temp/cmake-3.16.1

  
Another reference

`sudo apt-get install build-essential`  
`git clone https://github.com/tensorflow/tensorflow.git tensorflow_src`  
`cd tensorflow_src && ./tensorflow/lite/tools/make/download_dependencies.sh`  
  

