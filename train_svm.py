import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import joblib
from tqdm import tqdm

# Paths
dataset_path = "dataset"  # dataset/male , dataset/female

facenet = FaceNet()

X = []
y = []

# 🔥 Data augmentation function
def augment_image(img):
    augmented = []

    # Original
    augmented.append(img)

    # Flip
    augmented.append(cv2.flip(img, 1))

    # Brightness change
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    augmented.append(bright)

    # Slight blur
    blur = cv2.GaussianBlur(img, (5,5), 0)
    augmented.append(blur)

    return augmented


print("Loading dataset...")

for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)

    for file in tqdm(os.listdir(folder), desc=f"Processing {label}"):

        img_path = os.path.join(folder, file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        try:
            img = cv2.resize(img, (160,160))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 🔥 Augmentation
            images = augment_image(img)

            for aug_img in images:
                aug_img = np.expand_dims(aug_img, axis=0)

                embedding = facenet.embeddings(aug_img)

                X.append(embedding[0])
                y.append(label)

        except:
            continue


print(f"Total samples after augmentation: {len(X)}")

# 🔥 Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 🔥 Scale features (VERY important)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🔥 Better SVM
model = SVC(kernel='rbf', probability=True, C=10, gamma='scale')

print("Training SVM...")
model.fit(X, y)

# Save everything
joblib.dump(model, "gender_svm.pkl")
joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Training complete. Model saved.")