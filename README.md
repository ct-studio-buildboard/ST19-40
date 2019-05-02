
Summary:

This project consists of a software, algorithms for converting english letters in a particular font to ones in Greek. Using the training model, and
the font extractor this can also be extended to a range of languages. It includes code to run methods for learning font style transfer, as
well as a front end prototype that actually runs the full pipeline with a front end representation.

Runtime Instructions and parameters:

Prior to running:
Place the file from the link below in file in 'full_models' named as 'gen_D'
https://drive.google.com/file/d/1nAQziu6aOFaujHP_fSgf3LlHFBFvsHFN/view?usp=sharing

Running example frontend:
1. Install the requirements specified in 'requirements.txt'
2. In a terminal CD into the project folder and then into : "/primary_code"
3. From that terminal location run app.py
4. In chrome run http://127.0.0.1:5000/
5. Upload any jpg of font to see it's mapping in greek. There is a sample letter in the file. You can also try other
   jpg images with weird, but sometimes font like results :)

Running trainer:
1. Ensure you have cuda.gpu available
2. Install the requirements specified in 'requirements.txt'
3. Place your fonts in the relevant folder /fonts

Running generator:
2. Install the requirements specified in 'requirements.txt'
3. Run /primary_code/models/run_model_wrapper.py - it will just output numbers, but can be used for more stuff


Relevant Files and Folder:

/primary_code: Contains code used to run a general front end and run models

/primary_code/app.py: Contains code to run our UI and interface with the model

/font_extractor:contains code for font extraction

/font_extractor/font_extractor.py: code to extract fonts in arbitrary languages

/primary_code/models: Contains code used to train, and generate a model

/primary_code/models/helper_functions.py: support functions for visualizing and loading data

/primary_code/models/run_model_wrapper.py: contains code for running a preloaded generator model

/primary_code/models/train_unet_gan.py: contains code for training our model

/primary_code/models/unet_with_gan.py: contains code to generate a descriminator

/primary_code/comparison_model: Modification of code that did not generalize for doing style transfer mapping


Notes/Issues*:
    Please note the trainer code was tested on a linux environment, and the rest on mac and may not work properly with windows, or mac in the case of training.
