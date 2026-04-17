# Sound Recognition on Arduino Nano 33 BLE

This represents the submission for the second assignment of the course "Low-power embedded systems" of students:

- Riccardo Libanora
- Jacopo Scanavacca

# Instructions

The repository contains the files and instructions necessary for all three steps of the TinyML pipeline to deploy a minimal model for recognition of hand-generated sounds using microphone data on Arduino Nano 33 BLE

## 1. Data collection

Data gathering was done via the software [Audacity](https://www.audacityteam.org/).

By setting the sample rate to 16kHz and a mono audio channel, any PC microphone can be used to record data, while maintaining fidelity with the microphone on the Arduino device the model is deployed on.

The first results were obtained with just 1 minute of recorded samples for each class. Classes with high energy spikes like 'clap' and 'snap' were recorded at 60bpm, while constant low noises like 'rub' and 'silence' were recorded for minutes straight while continuously generating the noise. 

Additionally, for some classes we were able to procure additional training data from the public dataset "hands make sounds", from [99Sounds](https://99sounds.org/hands-make-sounds/)

## 2. Model training

This project is set up in a way that assumes you want to gather data on your own classes, hence why the previous step was reported here. However, we also made sure to upload the exact data we used for training in a separate [branch](https://github.com/rrikiliba/ArduinoNano33BLESoundRecognition/tree/dataset), if that's what floats your boat.

The class name assignment, as already mentioned, is fully automated, so you can add whatever class you want by just creating its training data.

The actual model training is done entirely within this Jupiter notebook [file](training/training.ipynb), which explains the steps pretty well on its own. If you don't know how to open it, you can either:

- open it in VS Code using the official [extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- open [Google Colab](https://colab.research.google.com/), go to File > Open notebook, select the GitHub tab and paste this repo's link

Once executed all the cells, the script will directly export the model to a C++ header file `gesture_recognition.h` and the mfcc constants in the file `mfcc_constants.h`, which will both be used in the next step. A `.tflite` file is also generated, but you can ignore it.

Make sure to generally follow the instructions in the notebook and, once you obtain your `.h` files, place them in this [folder](inference) (if the notebook is executed locally, they will be placed there automatically)

## 3. Inference

Once you have your C++ header files in the inference [folder](inference), you can open the folder as an Arduino Sketch, compile it and load it to the device via Arduino tools (such as Arduino IDE or Cloud).

Tuning into the serial output of the board will allow you to see inference for every detected sound. 