from django.http import JsonResponse
from rest_framework.decorators import api_view
import tensorflow as tf
import numpy as np
import pickle
import os
import json

# Path to the model and preprocessing tools
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")

# Function to load model and preprocessing tools
def load_model_and_tools():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
        return model, tokenizer, encoder
    except Exception as e:
        print(f"Error loading model or preprocessing tools: {e}")
        return None, None, None

@api_view(['POST'])
def predict_emotion(request):
    """
    API endpoint to predict emotion from text
    """
    try:
        # Load model and preprocessing tools
        model, tokenizer, encoder = load_model_and_tools()
        
        if model is None or tokenizer is None or encoder is None:
            return JsonResponse({"error": "Model or preprocessing tools not found. Please train the model first."}, status=500)
        
        # Get text from request
        data = json.loads(request.body)
        text = data.get("text", "")
        
        if not text:
            return JsonResponse({"error": "No text provided"}, status=400)
        
        # Preprocess text
        seq = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=50, padding='post', truncating='post')
        
        # Make prediction
        prediction = model.predict(padded)
        emotion_index = np.argmax(prediction)
        emotion = encoder.inverse_transform([emotion_index])[0]
        confidence = float(prediction[0][emotion_index])
        
        # Return prediction
        return JsonResponse({
            "text": text,
            "predicted_emotion": emotion,
            "confidence": confidence
        })
        
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
