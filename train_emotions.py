import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
import numpy as np
from datasets import load_from_disk
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Load GoEmotions dataset locally
# ===============================
print("Loading GoEmotions dataset from local disk...")
dataset = load_from_disk("go_emotions_dataset")

# Define the emotions we want to focus on
target_emotions = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

# Function to filter and process the dataset
def process_data(data):
    texts = []
    labels = []
    
    for item in data:
        text = item["text"]
        emotion_labels = [dataset["train"].features["labels"].feature.names[label] 
                          for label in item["labels"]]

        for emotion in target_emotions:
            if emotion in emotion_labels:
                texts.append(text)
                labels.append(emotion)
                break
    
    return texts, labels

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

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

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

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("target_emotions.pkl", "wb") as f:
    pickle.dump(target_emotions, f)

# ===============================
# Save training history as PNG table
# ===============================
print("Saving training history as PNG table...")

history_df = pd.DataFrame(history.history)
history_df["Epoch"] = history_df.index + 1
history_df = history_df[["Epoch", "accuracy", "loss", "val_accuracy", "val_loss"]]

# Round to 4 decimal places
history_df = history_df.round(4)

# Create table figure
fig, ax = plt.subplots(figsize=(8, 2 + 0.3 * len(history_df)))
ax.axis("off")
table = ax.table(cellText=history_df.values,
                 colLabels=history_df.columns,
                 cellLoc="center",
                 loc="center")

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.savefig("training_history_table.png", bbox_inches="tight")
plt.close()

print("Done! Model, preprocessing tools, and training history table saved.")
