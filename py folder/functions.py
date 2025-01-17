"""
1. Preprocessing Functions
Reusable functions for dataset preprocessing:
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the Galaxy Zoo dataset.
    
    Parameters:
        file_path (str): Path to the dataset file.
        
    Returns:
        X_train, X_test, y_train, y_test: Preprocessed data splits.
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Encode the 'class' column
    data['class_encoded'] = data['class'].astype('category').cat.codes
    
    # Normalize the 'redshift' column
    data['redshift_normalized'] = (data['redshift'] - data['redshift'].mean()) / data['redshift'].std()
    
    # Select features and target
    X = data[['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift_normalized']]
    y = data['class_encoded']
    
    # Split into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # One-hot encode the target labels
    y_train = to_categorical(y_train, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)
    
    return X_train, X_test, y_train, y_test

"""
2. Model Training Function
Generalize the model-building and training process:
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def build_and_train_model(X_train, y_train, X_test, y_test, activation_function, epochs=20, batch_size=32):
    """
    Build and train a Neural Network with a specific activation function.
    
    Parameters:
        X_train, y_train, X_test, y_test: Training and testing data.
        activation_function (str): Activation function for the hidden layers.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        
    Returns:
        model: The trained model.
        history: Training history object.
    """
    # Define the model
    model = Sequential([
        Dense(64, activation=activation_function, input_shape=(X_train.shape[1],)),
        Dense(32, activation=activation_function),
        Dense(3, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model, history

"""
3. Evaluation Functions
Reusable functions to evaluate and visualise model performance:
"""

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on the test set and generate performance metrics.
    
    Parameters:
        model: Trained model.
        X_test, y_test: Test data.
        
    Returns:
        metrics: A dictionary with accuracy, loss, and classification report.
    """
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Generate classification report
    class_report = classification_report(y_test_classes, y_pred_classes, output_dict=True)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
    
    # Visualize confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    return {"accuracy": test_accuracy, "loss": test_loss, "classification_report": class_report}

"""
Usage in Notebooks
Import these functions into your notebooks:
"""

from functions import load_and_preprocess_data, build_and_train_model, evaluate_model

# Example usage
X_train, X_test, y_train, y_test = load_and_preprocess_data("path_to_dataset.csv")
model, history = build_and_train_model(X_train, y_train, X_test, y_test, activation_function='relu')
metrics = evaluate_model(model, X_test, y_test)
