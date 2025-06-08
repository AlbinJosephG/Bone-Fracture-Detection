import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam

# Constants
IMG_SIZE = (224, 224)
MODEL_PATH = 'model/fracture_classifier.h5'

def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_fracture(model, image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)[0][0]

    label = "Fractured" if prediction > 0.5 else "Not Fractured"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")

    return img_array, label

def main(image_path, generate_gradcam=False):
    model = load_model(MODEL_PATH)

    img_array, label = predict_fracture(model, image_path)

    if generate_gradcam:
        # Change last_conv_layer_name based on your model
        last_conv_layer_name = 'conv2d_1'
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        save_and_display_gradcam(image_path, heatmap, cam_path='gradcam_output.jpg')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [--gradcam]")
        sys.exit(1)

    img_path = sys.argv[1]
    gradcam_flag = '--gradcam' in sys.argv

    main(img_path, generate_gradcam=gradcam_flag)
