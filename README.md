                                                         ** # Brain-Tumor-Classification-Using-CNN-on-MRI-Images**
**Project Overview**

 This project implements a deep learning model based on Convolutional Neural Networks (CNN) to classify brain tumors from MRI images. The model identifies three types of brain tumors — glioma, meningioma, and pituitary tumor — as well as no tumor cases, enabling automated and accurate diagnosis assistance.

Dataset
The dataset consists of MRI brain scans categorized into different tumor types:

Glioma Tumor

Meningioma Tumor

Pituitary Tumor

No Tumor

The images were preprocessed and organized into train and validation sets for model training.

Features
Preprocessing and image augmentation to improve model robustness

CNN architecture built with TensorFlow and Keras

Model evaluation with accuracy, precision, recall, and F1-score metrics

Visualization of training progress and confusion matrix for performance analysis

Easily extensible for transfer learning and fine-tuning with pretrained models

Results
The model achieved approximately 91% accuracy on the validation set with strong performance across all tumor classes, making it a promising tool for medical image classification tasks.

Metric	Score
Accuracy	91%
Precision	0.93 (weighted average)
Recall	0.91 (weighted average)
F1-score	0.90 (weighted average)

Usage
Clone the repository

Download and extract the MRI brain tumor dataset as per instructions

Run the Jupyter notebook or Python scripts for training and evaluation

Use the provided prediction scripts to test new MRI images

Future Work
Integrate transfer learning using pretrained models like MobileNetV2 or ResNet50

Implement real-time web app interface for interactive diagnosis

Extend to multi-modal medical imaging datasets

Requirements
Python 3.x

TensorFlow 2.x

Keras

Matplotlib

Scikit-learn

Seaborn
