# Image-Restoration-DL

This project aims to enhance low-light images using a deep learning model that combines convolutional and residual blocks, along with custom loss functions. The model is designed to improve the overall quality and visual clarity of photographs taken in low-light conditions.

## Table of Contents
- [Introduction](#introduction)
- [Architecture](#architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [References](#references)

## Introduction
Low-light conditions often degrade image quality, leading to poor visibility and high noise levels. This project implements a novel method to restore and enhance low-light images, achieving a high Peak Signal-to-Noise Ratio (PSNR) of 27.85.

## Architecture
The model architecture includes:
- Input Layer
- Convolutional Blocks (6 layers)
- Residual Blocks (5 layers)
- Deconvolutional Layers (2 layers)
- Output Layer

The model processes images through multiple stages of convolution, residual connections, and deconvolution to restore image quality.

## Training
The model is trained on the dataset, which contains 500 image pairs. The training process includes:
- Data preprocessing and augmentation
- Custom loss functions (Perceptual and Charbonnier losses)
- 50 epochs of training

## Evaluation
The model is evaluated using Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Mean Absolute Error (MAE). The evaluation script processes images and calculates these metrics to assess the model's performance.

## Results
- **MSE**: 107.18
- **PSNR**: 27.85
- **MAE**: 159.97

The results indicate a significant improvement in image quality, with a notable increase in PSNR compared to baseline models.

## Usage
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/low-light-image-restoration.git
    cd low-light-image-restoration
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Train the model:
    ```sh
    python train.py
    ```

4. Evaluate the model:
    ```sh
    python eval.py --images_path ./path_to_images --eval_path ./path_to_evaluation
    ```
