# 🩺 Pneumonia Detection System Using Deep Learning (ResNet50V2 & Gradio)

A complete end-to-end Deep Learning project for detecting Pneumonia from chest X-ray images. This repository contains the model training pipeline (`pneumonia_detection_training.ipynb`), the trained model assets, and an interactive web interface built with Gradio for real-time predictions.

---

## 🚀 Project Overview

Pneumonia is an inflammatory condition of the lung affecting primarily the microscopic air sacs. Early and accurate detection from chest X-rays is vital for effective medical intervention. This project leverages **Transfer Learning** using the pre-trained **ResNet50V2** architecture to classify chest X-ray images into two categories: **Normal** or **Pneumonia**.

---

## 📂 Repository Structure & Assets

You can check and review the core files and visual previews provided in this repository assets:
* **`pneumonia_detection_training.ipynb`**: Jupyter Notebook containing the complete dataset pipeline, model architecture, and training logic (with outputs cleared for clean version control).
* **`app.py`**: Gradio web application script designed for real-time inference and user interaction.
* **`requirements.txt`**: List of all required Python packages and dependencies to run the project.
* **`best_pneumonia_model.h5`**: The optimized trained model weights used by the Gradio app for accurate predictions.
* **`assets/dataset.png`**: Visual preview and overview of the dataset structure used for training. You can check the `assets` folder to view it.

---

## 📊 Dataset Information

* **Source Dataset:** Chest X-Ray Images (Pneumonia) sourced from Kaggle.
* **Kaggle Link:** [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* **Dataset Structure:** Divided into training and validation sets containing X-ray images belonging to **Normal** and **Pneumonia** classes.

---

## 🛠️ Tech Stack & Libraries

* **Python** (Programming Language)
* **TensorFlow / Keras** (Deep Learning Framework & ResNet50V2 Transfer Learning)
* **Gradio** (Interactive Web UI for model deployment)
* **NumPy & Matplotlib** (Data manipulation and visualization)
* **Google Colab & Google Drive** (Cloud training environment and model asset storage)

---

## 📊 Methodology & Model Pipeline

1. **Dataset Extraction:** The dataset is unzipped directly from Google Drive into the working environment.
2. **Data Preprocessing & Augmentation:** 
   * Images are resized to `224x224` pixels.
   * Pixel values are rescaled (`1.0 / 255`).
   * Data augmentation techniques (such as rotation range and horizontal flips) are applied to the training set to prevent overfitting.
3. **Transfer Learning (ResNet50V2):** 
   * The base model is loaded with pre-trained `ImageNet` weights with `trainable = False` to freeze feature extraction layers.
   * Custom dense classification layers (Global Average Pooling, Dense ReLU layer, and Sigmoid output layer) are appended on top.
4. **Callbacks & Optimization:** 
   * `EarlyStopping` is utilized to monitor validation AUC and prevent overfitting.
   * `ModelCheckpoint` automatically saves the best performing weights as `best_pneumonia_model.h5`.

---

## 💻 How to Run the Project

### 1. Clone the Repository
```bash
git clone [https://github.com/Nokhiz-Khan/Pneumonia-Detection-ResNet50V2.git](https://github.com/Nokhiz-Khan/Pneumonia-Detection-ResNet50V2.git)
cd Pneumonia-Detection-ResNet50V2
2. Install Dependencies
Make sure you have Python installed, then install the required packages:

Bash
pip install -r requirements.txt
3. Run the Training Notebook
Open the pneumonia_detection_training.ipynb file in Google Colab or Jupyter Notebook.

Connect your Google Drive containing your chest X-ray dataset and run the cells sequentially to reproduce the training process.

4. Launch the Web App
To run the interactive Gradio interface locally using the provided model weights, execute:

Bash
python app.py
🌟 Professional Highlights
Clean Code Standards: The Jupyter notebook is structured into logical, well-commented blocks (Setup, Dataset Extraction, Preprocessing, Model Architecture, and Gradio Interface) adhering to industry best practices.

Optimized Repository Size: Heavy training output logs are omitted from the notebook to keep the repository lightweight and version-control friendly.

Production-Ready UI: Integrated seamlessly with Gradio to allow users to upload custom X-ray images and instantly view confidence scores for Normal vs. Pneumonia classes.

📜 License
This project is open-source and available under the MIT License.