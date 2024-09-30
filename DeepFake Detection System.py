import os
import cv
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from flask import Flask, request, jsonify

# 1. Setup VGG16 model for feature extraction
vgg_model = VGG16(weights='imagenet', include_top=False)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = vgg_model.predict(img_data)
    return features

# 2. Data Collection and Preprocessing
def extract_frames(video_path, output_folder):
    cap = cv.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv.imwrite(os.path.join(output_folder, f"frame{count}.jpg"), frame)
        count += 1
    cap.release()

# Example paths (replace with actual paths)
real_video_path = 'real_video.mp4'
fake_video_path = 'fake_video.mp4'
real_output_folder = 'real_frames'
fake_output_folder = 'fake_frames'

os.makedirs(real_output_folder, exist_ok=True)
os.makedirs(fake_output_folder, exist_ok=True)

extract_frames(real_video_path, real_output_folder)
extract_frames(fake_video_path, fake_output_folder)

# 3. Feature Extraction and Labeling
def process_folder(folder, label):
    features = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img_features = extract_features(img_path)
            features.append(img_features.flatten())
            labels.append(label)
    return np.array(features), np.array(labels)

real_features, real_labels = process_folder(real_output_folder, 0)
fake_features, fake_labels = process_folder(fake_output_folder, 1)

# Combine and shuffle data
features = np.concatenate((real_features, fake_features))
labels = np.concatenate((real_labels, fake_labels))

indices = np.arange(features.shape[0])
np.random.shuffle(indices)
features = features[indices]
labels = labels[indices]

# 4. Model Training
model = Sequential()
model.add(Flatten(input_shape=(7, 7, 512)))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(features, labels, epochs=10, batch_size=32, validation_split=0.2)

# 5. Deployment using Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img_path = 'uploaded_image.jpg'
    file.save(img_path)
    features = extract_features(img_path)
    prediction = model.predict(features)
    result = 'fake' if prediction[0][0] > 0.5 else 'real'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
