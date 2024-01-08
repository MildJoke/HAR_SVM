import os
import numpy as np
from keras.preprocessing import image
from keras.applications import VGG19  # Import VGG19
from keras.applications.vgg19 import preprocess_input
import joblib

# Load the trained SVM model and label encoder
best_svm_model = joblib.load('best_svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize VGG19 model for feature extraction
base_model = VGG19(weights='imagenet', include_top=False)

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

def predict_activity(image_path, model, svm_model, label_enc):
    features = extract_features(image_path, model)
    features = features.reshape(1, -1)
    prediction = svm_model.predict(features)
    return label_enc.inverse_transform(prediction)[0]

def predict_activities_from_file(file_path, model, svm_model, label_enc):
    with open(file_path, 'r') as file:
        image_paths = file.readlines()

    predictions = [predict_activity(path.strip(), model, svm_model, label_enc) for path in image_paths]
    return predictions

# Path to the test file containing paths to test images
test_file_path = 'path_to_test_file.txt'

# Perform predictions
test_predictions = predict_activities_from_file(test_file_path, base_model, best_svm_model, label_encoder)

# Display predictions
for img_path, prediction in zip(open(test_file_path), test_predictions):
    print(f"Image: {img_path.strip()} - Prediction: {prediction}")
