from pathlib import Path
import random
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter
# import tensorflow as tf
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# tf.config.set_visible_devices([], 'GPU')
from torch import logit

interpreter = Interpreter(model_path="bubble_model_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# def preprocess_keras(image_path, target_size=128):

#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     h, w, _ = img.shape

#     # --- Resize preserving aspect ratio ---
#     scale = max(target_size / h, target_size / w)

#     new_h = int(h * scale)
#     new_w = int(w * scale)

#     img = cv2.resize(img, (new_w, new_h))

#     # --- Center crop ---
#     start_x = (new_w - target_size) // 2
#     start_y = (new_h - target_size) // 2

#     img = img[start_y:start_y+target_size, start_x:start_x+target_size]

#     img = img.astype(np.float32)

#     # MobileNet preprocessing
#     img = preprocess_input(img)

#     img = np.expand_dims(img, axis=0)

#     return img

def preprocess_image(image_path, target_size=128):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape

    # --- Resize while preserving aspect ratio ---
    scale = max(target_size / h, target_size / w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    img = cv2.resize(img, (new_w, new_h))

    # --- Center crop ---
    start_x = (new_w - target_size) // 2
    start_y = (new_h - target_size) // 2

    img = img[start_y:start_y+target_size, start_x:start_x+target_size]

    # Convert to float
    img = img.astype(np.float32)

    # --- MobileNetV2 preprocessing ---
    # img = (img / 127.5) - 1

    img = np.expand_dims(img, axis=0)

    return img


# def predict_keras(image_path):
#     model = tf.keras.models.load_model("bubble_classifier.keras")
#     image = predict_keras(image_path)
#     logit = model.predict(image)[0][0]

#     probability = 1 / (1 + np.exp(-logit))

#     print("Logit:", logit)
#     print("Probability:", probability)

#     if probability < 0.5:
#         result = "Marked"
#         confidence = (1 - probability) * 100
#     else:
#         result = "Unmarked"
#         confidence = probability * 100

#     return probability, result, confidence


def predict_bubble(image_path):
    # 1. Load image
    # raw_img = cv2.imread(image_path)
    # if raw_img is None: return "File not found"

    img = preprocess_image(image_path, target_size=128)

    # # 2. CRITICAL: Convert BGR (OpenCV default) to RGB (Keras default)
    # img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # # 3. Resize using Bilinear interpolation (Keras default)
    # img_resized = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_LINEAR)

    # # 4. Preprocess to exactly [-1, 1]
    # img_array = img_resized.astype(np.float32)
    # img_array = (img_array / 127.5) - 1.0
    
    # 5. Run Inference
    # input_data = np.expand_dims(img_array, axis=0)

    # print(f"Shape: {input_data.shape}")      # Should be (1, 128, 128, 3)
    # print(f"Dtype: {input_data.dtype}")      # Should be float32
    # print(f"Max pixel: {np.max(input_data)}") # Should be ~1.0
    # print(f"Min pixel: {np.min(input_data)}") # Should be ~-1.0

    interpreter.set_tensor(input_details[0]['index'], img)
    
    interpreter.invoke()
    
    # 6. Get Result
    output = interpreter.get_tensor(output_details[0]['index'])
    logit = output[0][0]
    # Sigmoid math for 0-1 probability
    # prob = output_data[0][0]
    probability = 1 / (1 + np.exp(-logit))

    print("Logit:", logit)
    print("Probability:", probability)
    
    if probability < 0.5:
        result = "Marked"
        confidence = (1 - probability) * 100
    else:
        result = "Unmarked"
        confidence = probability * 100

    return probability.item(), result, confidence
    
if __name__ == "__main__":
    input_image_folder = Path("dataset/unmarked")
    files = [f for f in input_image_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg"]]

    print(len(files), "files found")

    random_file = random.choice(files)
    print("Processing file:", random_file)

    print("Answer: ", predict_bubble(random_file))