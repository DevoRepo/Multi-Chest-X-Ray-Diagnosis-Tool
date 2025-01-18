# Multi-Chest-X-Ray-Diagnosis-Tool
## Chest X-Ray Classification with DenseNet121

This project demonstrates the application of machine learning in healthcare using a DenseNet121 model for the classification of chest X-ray images into four categories: Effusion, Atelectasis, Infiltration, and No Finding. The goal is to enhance the accuracy and efficiency of medical diagnosis by leveraging deep learning techniques.

## Overview

Machine learning, particularly deep learning, has the potential to revolutionize healthcare by providing rapid and accurate medical diagnoses. This project showcases how the DenseNet121 architecture can be used to classify chest X-ray images, aiding in the detection of various lung conditions.

## Features

- **Data Preprocessing**: Efficient preprocessing steps to map image paths, sanitize labels, and create binary columns for each condition.
- **Custom Data Generator**: Implementation of a custom data generator to handle image loading and preprocessing in batches, ensuring efficient data handling.
- **Model Design**: Utilization of the DenseNet121 architecture with pre-trained weights from ImageNet, fine-tuned for the specific task of chest X-ray classification.
- **Training and Evaluation**: Training the model on a balanced dataset, evaluating performance using ROC curves and AUC scores, and addressing data imbalance issues.

## Dataset

The dataset used in this project is a collection of chest X-ray images and associated metadata, including patient demographics and diagnostic labels. The data is sourced from a publicly available repository.

## Key Steps

1. **Data Preparation**:
   - Mapping image paths to ensure seamless access.
   - Processing patient information and sanitizing diagnostic labels.
   - Creating binary columns for each target condition.
   - Filtering and splitting the dataset into training, validation, and test sets.

2. **Custom Data Generator**:
   - Loading and preprocessing images in batches.
   - Resizing images to the required input size (224x224 pixels) and applying normalization.

3. **Model Design and Training**:
   - Building the DenseNet121 model with pre-trained weights and adding a global average pooling layer followed by a dense layer with sigmoid activation.
   - Compiling the model using the Adam optimizer and binary cross-entropy loss.
   - Training the model for 20 epochs with real-time validation.

4. **Evaluation**:
   - Making predictions on the test set and evaluating performance using ROC curves and AUC scores.
   - Checking label distribution in the test set to ensure sufficient class representation.
   - Providing additional evaluation metrics such as the confusion matrix and classification report.

## Results

The DenseNet121 model demonstrated robust performance in classifying the selected conditions, with high AUC scores for Effusion and No Finding. The balanced dataset approach helped reduce bias towards the more frequent `No_Finding` class, leading to more accurate and fair evaluation metrics.

