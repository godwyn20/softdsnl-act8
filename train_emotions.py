import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
import pickle

# Load GoEmotions dataset
print("Loading GoEmotions dataset...")
dataset = load_dataset("go_emotions")

# Define the emotions we want to focus on
target_emotions = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

# Function to filter and process the dataset
def process_data(data):
    texts = []
    labels = []
    
    for item in data:
        # Get the text
        text = item["text"]
        
        # Get the emotion labels
        emotion_labels = [dataset["train"].features["labels"].feature.names[label] for label in item["labels"]]
        
        # Check if any of our target emotions are in the labels
        for emotion in target_emotions:
            if emotion in emotion_labels:
                texts.append(text)
                labels.append(emotion)
                break
    
    return texts, labels

# Process the data
print("Processing training data...")
train_texts, train_labels = process_data(dataset["train"])
print("Processing validation data...")
val_texts, val_labels = process_data(dataset["validation"])
print("Processing test data...")
test_texts, test_labels = process_data(dataset["test"])

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Test samples: {len(test_texts)}")

# Encode labels
print("Encoding labels...")
encoder = LabelEncoder()
encoder.fit(target_emotions)
train_labels_encoded = encoder.transform(train_labels)
val_labels_encoded = encoder.transform(val_labels)
test_labels_encoded = encoder.transform(test_labels)

# Tokenize text
print("Tokenizing text...")
max_words = 10000
max_length = 50

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Build model
print("Building model...")
model = models.Sequential([
    layers.Embedding(max_words, 64, input_length=max_length),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(target_emotions), activation="softmax")
])

# Compile
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Print model summary
model.summary()

# Train the model
print("Training model...")
history = model.fit(
    train_padded, train_labels_encoded,
    epochs=10,
    validation_data=(val_padded, val_labels_encoded),
    batch_size=32
)

# Evaluate on test data
print("Evaluating model...")
test_loss, test_accuracy = model.evaluate(test_padded, test_labels_encoded)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save model + tokenizer + encoder
print("Saving model and preprocessing tools...")
model.save("emotion_model.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save encoder
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Save target emotions
with open("target_emotions.pkl", "wb") as f:
    pickle.dump(target_emotions, f)

print("Done! Model and preprocessing tools saved.")