from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import json
import io
from PIL import Image

app = FastAPI()

# Enable CORS so the browser allows the frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the model and class names
MODEL = tf.keras.models.load_model('models/eye_disease_model.keras')
with open('models/class_names.json', 'r') as f:
    CLASS_NAMES = json.load(f)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  
    return np.expand_dims(img_array, axis=0)

@app.get("/")
def home():
    return {"message": "Visionary AI Eye Disease Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    processed_img = preprocess_image(contents)
    
    predictions = MODEL.predict(processed_img)
    
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    # We use the raw prediction value as the confidence score
    confidence = float(np.max(predictions[0])) * 100

    return {
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)