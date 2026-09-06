# Pneumonia Detection using Chest X-Ray Images (ResNet50V2)

A deep learning project for detecting pneumonia from chest X-ray images using Transfer Learning with ResNet50V2, along with an interactive Gradio demo for real-time predictions.

## Overview

This project uses **Transfer Learning** to detect pneumonia from chest X-ray images:

- Pre-trained **ResNet50V2** (trained on ImageNet) is used as a frozen feature extractor
- Custom classification layers are added on top for binary classification (Normal vs Pneumonia)
- An interactive **Gradio** web app is included for live inference

## Dataset

The project uses the [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle, which contains:

- Training images of normal and pneumonia cases
- Validation set
- Test set

The dataset zip is stored in Google Drive and extracted at runtime (see Usage below).

## Model Architecture

**Transfer Learning with ResNet50V2**

- Base model: `ResNet50V2` pre-trained on ImageNet, with `include_top=False`
- Base layers frozen (`base_model.trainable = False`)
- `GlobalAveragePooling2D` on top of the base model output
- `Dense(128, activation='relu')` hidden layer
- `Dense(1, activation='sigmoid')` output layer for binary classification
- Optimizer: Adam
- Loss: Binary cross-entropy
- Metrics: Accuracy, AUC

## Training Setup

- Image size: `224 x 224`
- Batch size: `32`
- Data augmentation on training set: rotation, zoom, horizontal flip
- Validation data only rescaled (no augmentation)
- Callbacks:
  - `EarlyStopping` (monitors `val_auc`, patience = 3, restores best weights)
  - `ModelCheckpoint` (saves best model based on `val_auc`)
- Trained for up to 10 epochs

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
Pneumonia-Detection-ResNet50V2/
├── pneumonia_detection_training.ipynb   # Main training notebook (Colab)
├── app.py                               # Gradio demo app
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
└── assets/                              # Result images and demo screenshots
    ├── dataset_samples.png
    ├── training_results.png
    ├── gradio_demo_normal.png
    └── gradio_demo_pneumonia.png
```

## Usage

1. **Setup Environment**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Dataset**

   This project is built to run on **Google Colab**. Mount your Google Drive and place the dataset zip (`chest-xray-pneumonia.zip`) inside a `pneumonia_detection` folder in your Drive. The notebook will mount the drive and extract the dataset automatically:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Train the Model**

   Open `pneumonia_detection_training.ipynb` in Google Colab and run all cells in order:
   - Setup & imports
   - Dataset extraction
   - Data preprocessing & loaders
   - Model architecture (ResNet50V2)
   - Model training

   The best model is automatically saved to your Google Drive as `best_pneumonia_model.h5`.

4. **Run the Gradio Demo**

   ```bash
   python app.py
   ```

   This launches a local Gradio interface where you can upload a chest X-ray image and get a live "Normal vs Pneumonia" prediction with confidence scores.

## Results

Result visualizations (training curves and Gradio demo screenshots for both Normal and Pneumonia predictions) are available in the [`assets`](./assets) folder of this repository. Check there to see the model's performance and the demo in action.

## GPU Support

The training notebook is designed to run on Google Colab with GPU acceleration (T4 GPU recommended) for faster training.

## License

This project is for educational and research purposes.

## Acknowledgments

- Dataset: Paul Mooney (Kaggle)
- Pre-trained model: TensorFlow/Keras Applications (ResNet50V2)
