from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump, load
import mediapipe as mp
import numpy as np
import json
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Function to extract hand landmarks from an image
def extract_hand_landmarks(image_path):
    image = cv2.imread(str(image_path))
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        return np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in results.multi_hand_landmarks[0].landmark]).flatten()
    else:
        return None

# Define the path to the dataset directory
data_dir = Path('pathtodataset')
dataset_name = 'hagrid_120k'
dataset_path = data_dir / dataset_name

# Separate annotation directory and image directory
annotation_dir = (dataset_path / 'ann_train_val')
img_dir = [d for d in dataset_path.iterdir() if d.is_dir() and 'ann_train_val' not in str(d)][0]

# Initialize lists to store image paths, labels, and annotations
image_paths = []
labels = []
annotations = {}

# Load annotations from JSON files
for annotation_file in annotation_dir.iterdir():
    with open(annotation_file, 'r') as file:
        annotation_data = json.load(file)
        annotations.update(annotation_data)
for class_folder in img_dir.iterdir():
    class_label = class_folder.name.split('_')[-1]
    for image_file in class_folder.iterdir():
        image_paths.append(image_file)
        labels.append(class_label)
X = []
y = []
for image_path, label in zip(image_paths, labels):
    landmarks = extract_hand_landmarks(image_path)
    if landmarks is not None:
        X.append(landmarks)
        y.append(label)
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Save the trained model
model_path = 'hand_gesture_model.joblib'
dump(model, model_path)
