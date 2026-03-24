import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier

MODEL_PATH = "models/knn_model.pkl"

def train_knn_dummy():
    # Create simple dummy data: 2 classes
    # Class 0: values near 0
    X_class0 = np.random.normal(loc=0.0, scale=0.5, size=(50, 2))
    y_class0 = np.zeros(50)
    
    # Class 1: values near 5
    X_class1 = np.random.normal(loc=5.0, scale=0.5, size=(50, 2))
    y_class1 = np.ones(50)
    
    X = np.vstack((X_class0, X_class1))
    y = np.concatenate((y_class0, y_class1))
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(knn, f)
        
    return "Model trained successfully with dummy data!"

def classify_dummy_feature(feature_vector):
    if not os.path.exists(MODEL_PATH):
        return "Model not found. Please train it first."
        
    with open(MODEL_PATH, 'rb') as f:
        knn = pickle.load(f)
        
    feature_vector = np.array(feature_vector).reshape(1, -1)
    prediction = knn.predict(feature_vector)
    
    return f"Predicted Class: {int(prediction[0])}"
