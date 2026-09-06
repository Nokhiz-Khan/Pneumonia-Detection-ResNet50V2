
import gradio as gr
import numpy as np
import tensorflow as tf

# 1. Load trained model weights
model = tf.keras.models.load_model('best_pneumonia_model.h5')


# 2. Main prediction function
def predict_pneumonia(input_image):
  # Resize and normalize image
  img = input_image.resize((224, 224))
  img_array = np.array(img) / 255.0
  img_array = np.expand_dims(img_array, axis=0)

  # Model inference
  prediction = model.predict(img_array)[0][0]
  pneumonia_prob = float(prediction)
  normal_prob = float(1.0 - prediction)

  return {'Normal': normal_prob, 'Pneumonia': pneumonia_prob}


# 3. Build UI
interface = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(type='pil', label='Upload Chest X-Ray'),
    outputs=gr.Label(num_top_classes=2, label='Prediction Results'),
    title='Pneumonia Detection System',
)

if __name__ == '__main__':
  interface.launch()
