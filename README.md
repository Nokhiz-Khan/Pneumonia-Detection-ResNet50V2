# Pneumonia Detection from Chest X-Rays using ResNet50V2

An end-to-end deep learning solution designed to detect Pneumonia from Chest X-Ray images using Transfer Learning with ResNet50V2 and an interactive web deployment via Gradio.

## Key Features
- **Architecture:** Pre-trained ResNet50V2 base model with custom dense layers.
- **Regularization:** Early Stopping (`patience=3`) and Model Checkpointing to prevent overfitting.
- **Web Interface:** Interactive Gradio UI for quick image uploads and probabilistic output display.

## Repository Structure
- `app.py`: Gradio Web UI implementation script.
- `requirements.txt`: Python dependencies required for execution.
- `assets/`: Directory containing performance visual documentation, evaluation logs, and sample UI demonstrations.

## Documentation & Assets
All experimental visualizations and UI proofs are documented inside the `assets/` directory:
- Sample Dataset Preview: `assets/dataset_samples.png`
- Epoch Performance Logs: `assets/training_results.png`
- UI Demo (Pneumonia Detection): `assets/gradio_demo_pneumonia.png`
- UI Demo (Normal Detection): `assets/gradio_demo_normal.png`

## How to Run Locally
1. Clone repository:
   ```bash
   git clone [https://github.com/Nokhiz-Khan/Pneumonia-Detection-ResNet50V2.git](https://github.com/Nokhiz-Khan/Pneumonia-Detection-ResNet50V2.git)
   cd Pneumonia-Detection-ResNet50V2

## Install dependencies:

pip install -r requirements.txt

## Run the Gradio app:

python app.py
