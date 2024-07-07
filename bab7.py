import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
file_path = 'Rockpaper.csv'
data = pd.read_csv(file_path)

# Drop 'Unnamed: 0' column if exists
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

# Select features and target variable
X = data[['Score']]  # Features
y = data['TeamName']  # Target variable

# Encode target variable (TeamName)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build SVM model with linear kernel
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set: {accuracy:.2f}')

# Function to plot SVM decision boundaries
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    
    # Create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30).reshape(-1, 1)
    
    # Predictions on grid
    y_pred = model.predict(x)
    
    # Plot decision boundary
    ax.plot(x, y_pred, color='k', alpha=0.5, linestyle='--', label='Decision Boundary')
    
    # Plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], np.zeros_like(model.support_vectors_[:, 0]),
                   s=100, linewidth=1, facecolors='none', edgecolors='r', label='Support Vectors')
    
    ax.set_xlim(xlim)
    ax.set_ylim(min(y_pred) - 1, max(y_pred) + 1)

# Plot the SVM decision boundary with support vectors
plt.figure(figsize=(10, 6))
plt.scatter(X_train_scaled, y_train, c=y_train, s=50, cmap='viridis')
plot_svc_decision_function(svm_model)
plt.title('SVM Decision Boundary with Support Vectors')
plt.xlabel('Score (scaled)')
plt.ylabel('TeamName (encoded)')
plt.legend()
plt.grid(True)
plt.show()
