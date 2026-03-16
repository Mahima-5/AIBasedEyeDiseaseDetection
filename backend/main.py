from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import json
import io
from PIL import Image

app = FastAPI()

# 1. Load the model and class names
# Make sure the paths match where you saved your model earlier
MODEL = tf.keras.models.load_model('models/eye_disease_model.keras')
with open('models/class_names.json', 'r') as f:
    CLASS_NAMES = json.load(f)

def preprocess_image(image_bytes):
    # Preprocess the image (same pipeline used in training) [cite: 51]
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalization
    return np.expand_dims(img_array, axis=0)

@app.get("/")
def home():
    return {"message": "Visionary AI Eye Disease Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Accept the uploaded image [cite: 50]
    contents = await file.read()
    processed_img = preprocess_image(contents)
    
    # Load the trained model and predict [cite: 52, 53]
    predictions = MODEL.predict(processed_img)
    
    # Get the class with the highest probability
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(predictions[0])) * 100

    # Return the JSON-based API response [cite: 59]
    return {
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }

if __name__ == "__main__":
    import uvicorn
    # Start the FastAPI server [cite: 29]
    uvicorn.run(app, host="0.0.0.0", port=8000)