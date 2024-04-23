# Sleep-Stages-Classification-sleep-edf
 Sleep Stages Classification with PyTorch. Using Physionet Sleep edf dataset

First 20 Subjects are used for this project. 

Leave-one-Subject-out (LOSO) cross-validation is performed. Thus, one subject is choosen as test data, another subject is choosen as validation data, while remaining 18 are used as training data.

150 epochs are used during the training with Early Stopping technique, where patience were set to 15.

Implementation is based on [this code](https://github.com/emadeldeen24/AttnSleep.git)