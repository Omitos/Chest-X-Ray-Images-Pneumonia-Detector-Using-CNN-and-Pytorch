# Chest-X-Ray-Images-Pneumonia-Detector-Using-CNN-and-Pytorch

This repository contains a deep learning project for classifying chest X-ray images into two categories: **PNEUMONIA** and **NORMAL**. The model is based on EfficientNetV2 (with other options like CNN, ResNet34, and Fully Connected models), and it's implemented using PyTorch, torchvision, timm, and albumentations for data augmentation.

## Project Overview

The goal of this project is to train a deep learning model capable of accurately classifying chest X-ray images from the provided dataset. The dataset consists of images belonging to two classes:

- **PNEUMONIA**
- **NORMAL**

The data is split into training, validation, and testing sets. Data augmentation techniques are applied to the training data to improve the generalization of the model.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- timm
- albumentations
- matplotlib
- tqdm
- scikit-learn
- numpy

You can install the necessary dependencies using:

```bash
pip install -r requirements.txt
```
```bash
.
├── chest_xray/
│   ├── train/
│   │   ├── PNEUMONIA/
│   │   └── NORMAL/
│   └── test/
│       ├── PNEUMONIA/
│       └── NORMAL/
├── train_model.py        # Main script for training the model
├── utils.py              # Utility functions for data loading, preprocessing, and evaluation
├── README.md             # This file
└── requirements.txt      # Python package dependencies
```
```css
chest_xray/
├── train/
│   ├── PNEUMONIA/
│   └── NORMAL/
└── test/
    ├── PNEUMONIA/
    └── NORMAL/
````
## Data Augmentation
The following data augmentations are applied to the training dataset using the albumentations library:

Rotation (limit: 20 degrees)
Horizontal flip
Color jitter (brightness, contrast, saturation, hue)
Shift, scale, and rotate
Perspective transformation
For validation and test datasets, only normalization and resizing are applied.

## Splitting Data
The dataset is split into training and validation sets based on patient IDs. To avoid data leakage, the split is done such that images from the same patient appear in either the training or validation set, but not both.

## Model Architecture
Available Models:
EfficientNetV2 (default model)
CNN (Convolutional Neural Network)
ResNet34
Fully Connected (FC)
You can select the model type by changing the h["model"] hyperparameter.

## Training
The model is trained using the following hyperparameters:

Batch size: 256
Number of epochs: 10
Learning rate: 0.001
Early stopping: Activated with a patience threshold (default: infinite)
Scheduler: Cosine Annealing LR
Weight balancing: Optional, can be enabled using the h["balance"] parameter
The training process also includes early stopping, which halts training if the validation loss doesn't improve after a specified number of epochs.

## Evaluation
The model's performance is evaluated using the following metrics:

Loss
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
The evaluation results are plotted using matplotlib.

## Usage
Prepare the dataset: Ensure that the dataset is structured as mentioned earlier in the chest_xray/ folder.
Configure hyperparameters: Adjust the hyperparameters in the h dictionary, such as batch size, learning rate, and model type.
Train the model: Run the train_model.py script to train the model.
Evaluate the model: Once training is complete, the model is evaluated on the test set, and performance metrics are displayed.
Plot metrics: The script will generate plots for the training and validation loss over epochs and a confusion matrix for the test set.

## Results
Once the model is trained, it will output the following information:

Training and validation loss curves
Test accuracy and loss
Classification metrics (Precision, Recall, F1 Score)
Confusion Matrix
Acknowledgements
The project uses the Chest X-Ray Pneumonia Dataset from Kaggle.
This project leverages the EfficientNetV2 model from the timm library and utilizes PyTorch for training and evaluation.
