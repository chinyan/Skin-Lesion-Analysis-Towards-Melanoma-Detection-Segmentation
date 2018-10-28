# Our approach to tackle ISIC 2017 Skin Lesion Segmentation

More info on this Kaggle competition can be found on https://challenge.kitware.com/#phase/5841916ccad3a51cc66c8db0

This deep neural network achieved Jaccard Index of 0.747 and Dice Coefficient of 0.834.

The architecture was inspired by U-Net: Convolutional Networks for Biomedical Image Segmentation. We used the code provided by RECOD Titans as a starting point.

## Overview
### Data
Provided data is processed by ISIC_dataset.py script. 

### Pre-processing
The images are resized to 128x128. Data augmentations such as random rotation, horizontal flip and vertical flip are added at runtime by using ImageDataGenerator module. 

### Model
Train the model and generate masks for validation and test images
Run python segment.py to train the model.
Check out segment.py to modify the number of iterations (epochs), batch size, etc.

### Prediction
Run performance.py to get performance on validation and test images.
