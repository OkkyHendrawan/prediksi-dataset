import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
file_path = 'Rockpaper.csv'
data = pd.read_csv(file_path)

# Drop 'Unnamed: 0' column if exists
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

# Select features and target variable
X = data[['Score']].values  # Features as numpy array
y = data['TeamName'].values  # Target variable as numpy array

# Encode target variable (TeamName)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Limit data to first 50 teams
X_first_50 = X[:50]
y_encoded_first_50 = y_encoded[:50]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_first_50, y_encoded_first_50, test_size=0.2, random_state=42)

# Define the ANFIS model using skfuzzy
class ANFIS:
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(10, input_shape=(self.n_inputs,), activation='relu'))
        model.add(Dense(self.n_outputs, activation='softmax'))  # Use softmax for multi-class classification
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs=100):
        self.model.fit(X_train, tf.keras.utils.to_categorical(y_train, num_classes=self.n_outputs), epochs=epochs, verbose=0)

    def predict(self, X_test):
        return self.model.predict(X_test)

# Create and train Neuro Fuzzy System (ANFIS)
anfis = ANFIS(n_inputs=1, n_outputs=len(label_encoder.classes_))
anfis.train(X_train, y_train, epochs=100)  # Use categorical_crossentropy for multi-class classification

# Predictions on test set
y_pred = anfis.predict(X_test)

# Convert softmax probabilities to class predictions
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Accuracy on test set: {accuracy:.2f}')

# Sort test data for better visualization
sort_indices = np.argsort(X_test.flatten())
X_test_sorted = X_test[sort_indices]
y_test_sorted = y_test[sort_indices]
y_pred_sorted = y_pred_classes[sort_indices]

# Visualization
plt.figure(figsize=(12, 8))
plt.scatter(X_test_sorted, label_encoder.inverse_transform(y_test_sorted), color='blue', label='Actual', alpha=0.6)
plt.scatter(X_test_sorted, label_encoder.inverse_transform(y_pred_sorted), color='red', marker='x', label='Predicted', alpha=0.8)
plt.plot(X_test_sorted, label_encoder.inverse_transform(y_pred_sorted), color='red', linestyle='-', alpha=0.3, label='Trend Line')
plt.title('Neuro Fuzzy System Predictions (First 50 Teams)')
plt.xlabel('Score')
plt.ylabel('TeamName')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
