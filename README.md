# Landsat Cloud Removal Project

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Data Processing Pipeline](#data-processing-pipeline)
7. [Model Architecture](#model-architecture)
8. [Training](#training)
9. [Evaluation](#evaluation)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

This project implements a cloud removal pipeline for Landsat satellite imagery using a Convolutional Transformer model. The pipeline includes data downloading, preprocessing, model training, and evaluation stages.

## Project Structure

```
landsat-cloud-removal/
│
├── data/
│   ├── geometry/
│   ├── landsat_dataset/
│   └── train_valid/
│
├── models/
│
├── src/
│   ├── config/
│   │   └── settings.py
│   ├── data/
│   │   ├── download.py
│   │   └── processing.py
│   ├── models/
│   │   └── conv_transformer.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/taikiogihara/landsat-cloud-removal.git
   cd landsat-cloud-removal
   ```

2. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Configuration

The project configuration is managed through the `settings.py` file located in `src/config/`. You can modify the settings directly in this file or use environment variables to override the default values.

Key configuration options include:

- `PREFECTURE`: The target prefecture for data collection (default: "岡山県")
- `YEAR`: The year for which to fetch Landsat data (default: 2023)
- `TILE_SIZE`: Size of image tiles for processing (default: 256)
- `N_SAMPLE`: Number of samples to use for training (default: 1000)
- `EPOCHS`: Number of training epochs (default: 1000)
- `BATCH_SIZE`: Batch size for training (default: 16)
- `DEPTH`: Depth of the Transformer model (default: 3)

For a complete list of configuration options, refer to the `settings.py` file.

## Usage

To run the complete pipeline, use the following command:

```
cd src
python3 main.py
```

You can also run individual components of the pipeline:

1. Data download:
   ```
   python3 src/download.py
   ```

2. Data processing:
   ```
   python3 src/processing.py
   ```

3. Model training:
   ```
   python3 src/conv_transformer.py
   ```

## Data Processing Pipeline

The data processing pipeline consists of the following steps:

1. **Download geometry data**: Fetches and processes prefecture boundary data.
2. **Fetch Landsat data**: Queries the Landsat API for available imagery based on the specified prefecture and year.
3. **Download Landsat images**: Downloads the selected Landsat images and their metadata.
4. **Create masks**: Generates cloud and shadow masks from the QA pixel data.
5. **Process images**: Rotates and crops the images to a standard size.
6. **Select training dataset**: Chooses images for the training set based on cloud coverage.
7. **Create tiles**: Divides the processed images into tiles for model input.

## Model Architecture

The cloud removal model uses a Convolutional Transformer architecture, which combines the strengths of Convolutional Neural Networks (CNNs) and Transformer models. The main components are:

1. **Convolutional layers**: Initial feature extraction
2. **Transformer blocks**: Self-attention mechanism for capturing long-range dependencies
3. **Deconvolutional layers**: Upsampling and final image reconstruction

The model is implemented in the `ConvTransformer` class in `src/conv_transformer.py`.

## Training

The model is trained using the following process:

1. Data is loaded and preprocessed using the `create_datasets` function.
2. The ConvTransformer model is instantiated with the specified hyperparameters.
3. The model is compiled with the Nadam optimizer and a custom MaskedMSELoss.
4. Training is performed using `model.fit()` with early stopping and model checkpointing.
5. The best model and weights are saved for later use.

Training progress can be monitored using TensorBoard. To start TensorBoard, run:

```
tensorboard --logdir=models/[EXPERIMENT_NAME]/logs
```

## Evaluation

Model evaluation is performed on a held-out validation set. The evaluation metrics include:

- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)

Additionally, visual comparisons between the input cloudy images, ground truth clear images, and model predictions are generated for qualitative assessment.
