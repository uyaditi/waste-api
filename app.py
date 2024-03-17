from flask import Flask, request, jsonify
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("classifyWaste.h5")

output_class = ["biodegradable", "e-waste", "medical", "recyclable-glass",
                "recyclable-metal", "recyclable-paper", "recyclable-plastic"]


# def waste_prediction(new_image):
#     test_image = image.load_img(new_image, target_size=(224, 224))
#     test_image = image.img_to_array(test_image) / 255
#     test_image = np.expand_dims(test_image, axis=0)
#     predicted_array = model.predict(test_image)
#     predicted_value = output_class[np.argmax(predicted_array)]
#     predicted_accuracy = round(np.max(predicted_array) * 100, 2)
#     return predicted_value, predicted_accuracy

def waste_prediction(file):
    # Load image from file object
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize image
    # Expand dimensions for model input
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predicted_array = model.predict(img_array)
    predicted_value = output_class[np.argmax(predicted_array)]
    predicted_accuracy = round(np.max(predicted_array) * 100, 2)

    return predicted_value, predicted_accuracy


@app.route('/predict', methods=['POST'])
def predict_waste():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        predicted_value, predicted_accuracy = waste_prediction(file)
        return jsonify({'waste_material': predicted_value, 'accuracy': predicted_accuracy})


if __name__ == '__main__':
    app.run(debug=True)
