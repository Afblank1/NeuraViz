from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import json
import io
import uuid
import re
from datetime import datetime
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import warnings
import logging
import traceback
import math
warnings.filterwarnings('ignore')

# Configure comprehensive logging
logger = logging.getLogger("NeuraViz")


# Create file handler
file_handler = logging.FileHandler("neuraviz_training.log")
file_handler.setLevel(logging.INFO)

# Create console handler  
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger


# Initial startup messages


app = FastAPI(title="NeuraViz Backend", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests
class LayerConfig(BaseModel):
    layer_type: str
    params: Dict[str, Any]
    activation: Optional[str] = None
    dropout_rate: Optional[float] = None
    batch_norm: Optional[bool] = None
    learning_rate: Optional[float] = None
    weight_init: Optional[str] = None
    use_bias: Optional[bool] = None

class NetworkConfig(BaseModel):
    name: Optional[str] = "My Neural Network"
    layers: List[LayerConfig]
    optimizer: str = "adam"
    learning_rate: float = 0.001
    loss_function: str = "mse"
    batch_size: int = 32
    epochs: int = 100

class TrainingRequest(BaseModel):
    network_config: NetworkConfig
    dataset_name: Optional[str] = None
    custom_data: Optional[Dict[str, Any]] = None
    # Data split configuration
    train_split: float = 0.7
    validation_split: float = 0.2
    test_split: float = 0.1
    use_test_set: bool = True
    random_seed: int = 42

def sanitize_model_name(name: str) -> str:
    """Convert user-provided network name to a valid model ID"""
    if not name or not name.strip():
        return f"model_{uuid.uuid4().hex[:8]}"
    
    # Remove special characters and replace spaces with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9\s\-_]', '', name.strip())
    sanitized = re.sub(r'\s+', '_', sanitized)
    
    # Ensure it doesn't start with a number or special char
    if sanitized and not sanitized[0].isalpha():
        sanitized = f"model_{sanitized}"
    
    # Limit length and ensure uniqueness
    sanitized = sanitized[:50]  # Reasonable length limit
    
    # If empty after sanitization, use fallback
    if not sanitized:
        return f"model_{uuid.uuid4().hex[:8]}"
        
    return sanitized

def sanitize_value(value):
    """Convert non-JSON-compliant values to JSON-safe ones"""
    if isinstance(value, float):
        if math.isinf(value):

            return None if value < 0 else 1e308  # -inf -> null, +inf -> large number
        elif math.isnan(value):

            return None
    return value

def sanitize_recursively(obj):
    """Recursively sanitize all values in nested structures"""
    if isinstance(obj, dict):
        return {k: sanitize_recursively(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_recursively(item) for item in obj]
    else:
        return sanitize_value(obj)

def serialize_model_data(model_data: Dict) -> Dict:
    """
    Create a JSON-serializable version of model data by excluding PyTorch objects
    and other non-serializable items.
    """
    
    serializable_data = {}
    
    # List of keys that should be excluded from JSON serialization
    exclude_keys = {'model'}  # PyTorch model objects
    
    for key, value in model_data.items():
        if key not in exclude_keys:
            # Handle tensor objects if any exist in the future
            if hasattr(value, 'tolist'):  # numpy arrays or tensors
                serializable_data[key] = sanitize_recursively(value.tolist())
            else:
                serializable_data[key] = sanitize_recursively(value)
    
    return serializable_data

def serialize_all_models(models: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Create a JSON-serializable version of all models data.
    """
    return {model_id: serialize_model_data(model_data) 
            for model_id, model_data in models.items()}

# In-memory storage (in production, use a database)
active_models: Dict[str, Dict] = {}
datasets: Dict[str, Dict] = {}

def initialize_datasets():
    """Initialize comprehensive dataset collection for neural network training"""
    from sklearn.datasets import (
        make_regression, make_classification, load_iris, load_wine,
        load_breast_cancer, make_blobs, make_circles, make_moons,
        load_digits, fetch_california_housing, load_diabetes
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST, CIFAR10
    

    
    # === REAL REGRESSION DATASETS ===
    
    # 1. California Housing Dataset (Real)

    try:
        california = fetch_california_housing()
        
        datasets["california_housing"] = {
            "name": "California Housing Prices (Real)",
            "type": "regression",
            "category": "real_estate",
            "X": california.data.tolist(),
            "y": california.target.tolist(),
            "input_size": california.data.shape[1],
            "output_size": 1,
            "total_samples": len(california.data),
            "description": "Real California housing prices from 1990 census data",
            "features": california.feature_names,
            "target": "median_house_value_in_100k",
            "sample_count": len(california.data),
            "data_source": "California 1990 Census"
        }

    except Exception as e:
        pass
    
    # 2. Diabetes Dataset (Real)

    try:
        diabetes = load_diabetes()
        
        datasets["diabetes"] = {
            "name": "Diabetes Progression (Real)",
            "type": "regression",
            "category": "medical",
            "X": diabetes.data.tolist(),
            "y": diabetes.target.tolist(),
            "input_size": diabetes.data.shape[1],
            "output_size": 1,
            "total_samples": len(diabetes.data),
            "description": "Real medical data: predict diabetes progression after 1 year",
            "features": diabetes.feature_names,
            "target": "disease_progression_score",
            "sample_count": len(diabetes.data),
            "data_source": "Medical research study"
        }

    except Exception as e:
        pass
    
    
    
    # 3. Digits Dataset (Real)

    try:
        digits = load_digits()
        X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        
        datasets["digits"] = {
            "name": "Handwritten Digits (Real)",
            "type": "classification",
            "category": "image",
            "X_train": X_train_digits.tolist(),
            "X_test": X_test_digits.tolist(),
            "y_train": y_train_digits.tolist(),
            "y_test": y_test_digits.tolist(),
            "input_size": digits.data.shape[1],  # 64 features (8x8 images)
            "output_size": 10,
            "description": "Real 8x8 handwritten digit images (0-9)",
            "features": [f"pixel_{i}" for i in range(64)],
            "classes": [str(i) for i in range(10)],
            "image_shape": [8, 8, 1],
            "sample_count": len(digits.data),
            "data_source": "UCI ML Repository"
        }

    except Exception as e:
        pass
    
    # === REAL IMAGE DATASETS FOR CNN ===
    
    # 4. Real MNIST Subset

    try:
        # Load a real subset of MNIST
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_dataset = MNIST(root='/tmp/mnist', train=True, download=True, transform=transform)
        
        # Create a balanced subset (100 samples per class = 1000 total)
        samples_per_class = 100
        X_mnist_real = []
        y_mnist_real = []
        
        class_counts = {i: 0 for i in range(10)}
        
        for img, label in mnist_dataset:
            if class_counts[label] < samples_per_class:
                # Convert tensor to numpy and flatten
                img_array = img.numpy().flatten()
                X_mnist_real.append(img_array)
                y_mnist_real.append(label)
                class_counts[label] += 1
                
                # Stop when we have enough samples
                if all(count >= samples_per_class for count in class_counts.values()):
                    break
        
        X_mnist_real = np.array(X_mnist_real)
        y_mnist_real = np.array(y_mnist_real)
        
        X_train_mnist_real, X_test_mnist_real, y_train_mnist_real, y_test_mnist_real = train_test_split(
            X_mnist_real, y_mnist_real, test_size=0.2, random_state=42, stratify=y_mnist_real
        )
        
        datasets["mnist_real"] = {
            "name": "MNIST Digits (Real Subset)",
            "type": "classification",
            "category": "image",
            "X_train": X_train_mnist_real.tolist(),
            "X_test": X_test_mnist_real.tolist(),
            "y_train": y_train_mnist_real.tolist(),
            "y_test": y_test_mnist_real.tolist(),
            "input_size": 784,  # 28x28 flattened
            "output_size": 10,
            "description": "Real MNIST handwritten digits (28x28) - subset of 1000 samples per class",
            "image_shape": [28, 28, 1],
            "classes": [str(i) for i in range(10)],
            "sample_count": len(X_mnist_real),
            "data_source": "MNIST Database"
        }

    except Exception as e:
        pass

    
    # 5. Real CIFAR-10 Subset  

    try:
        # Load a real subset of CIFAR-10
        cifar_dataset = CIFAR10(root='/tmp/cifar10', train=True, download=True, transform=transform)
        
        # Create a smaller balanced subset (50 samples per class = 500 total)
        samples_per_class = 50
        X_cifar_real = []
        y_cifar_real = []
        
        class_counts = {i: 0 for i in range(10)}
        cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        for img, label in cifar_dataset:
            if class_counts[label] < samples_per_class:
                # Convert tensor to numpy and flatten
                img_array = img.numpy().flatten()
                X_cifar_real.append(img_array)
                y_cifar_real.append(label)
                class_counts[label] += 1
                
                # Stop when we have enough samples
                if all(count >= samples_per_class for count in class_counts.values()):
                    break
        
        X_cifar_real = np.array(X_cifar_real)
        y_cifar_real = np.array(y_cifar_real)
        
        X_train_cifar_real, X_test_cifar_real, y_train_cifar_real, y_test_cifar_real = train_test_split(
            X_cifar_real, y_cifar_real, test_size=0.2, random_state=42, stratify=y_cifar_real
        )
        
        datasets["cifar10_real"] = {
            "name": "CIFAR-10 Objects (Real Subset)",
            "type": "classification",
            "category": "image",
            "X_train": X_train_cifar_real.tolist(),
            "X_test": X_test_cifar_real.tolist(),
            "y_train": y_train_cifar_real.tolist(),
            "y_test": y_test_cifar_real.tolist(),
            "input_size": 3072,  # 32x32x3 flattened
            "output_size": 10,
            "description": "Real CIFAR-10 object images (32x32 RGB) - Balanced subset",
            "image_shape": [32, 32, 3],
            "classes": cifar_classes,
            "sample_count": len(X_cifar_real),
            "data_source": "CIFAR-10 Database"
        }

    except Exception as e:
        pass

    
    # === CLASSIC ML DATASETS ===
    
    # 1. Iris Dataset (already existing, enhanced)
    iris = load_iris()
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    datasets["iris"] = {
        "name": "Iris Dataset",
        "type": "classification",
        "category": "classic",
        "X_train": X_train_iris.tolist(),
        "X_test": X_test_iris.tolist(),
        "y_train": y_train_iris.tolist(),
        "y_test": y_test_iris.tolist(),
        "input_size": iris.data.shape[1],
        "output_size": 3,
        "description": "Classic flower species classification",
        "features": ["sepal length", "sepal width", "petal length", "petal width"],
        "classes": ["setosa", "versicolor", "virginica"]
    }
    
    # 2. Wine Quality Dataset
    wine = load_wine()
    X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42
    )
    
    datasets["wine_quality"] = {
        "name": "Wine Quality Dataset",
        "type": "classification",
        "category": "classic",
        "X_train": X_train_wine.tolist(),
        "X_test": X_test_wine.tolist(),
        "y_train": y_train_wine.tolist(),
        "y_test": y_test_wine.tolist(),
        "input_size": wine.data.shape[1],
        "output_size": 3,
        "description": "Wine classification based on chemical analysis",
        "features": wine.feature_names,
        "classes": wine.target_names.tolist()
    }
    
    # 3. Breast Cancer Dataset (Medical)
    cancer = load_breast_cancer()
    X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )
    
    datasets["breast_cancer"] = {
        "name": "Breast Cancer Wisconsin",
        "type": "classification",
        "category": "medical",
        "X_train": X_train_cancer.tolist(),
        "X_test": X_test_cancer.tolist(),
        "y_train": y_train_cancer.tolist(),
        "y_test": y_test_cancer.tolist(),
        "input_size": cancer.data.shape[1],
        "output_size": 2,
        "description": "Binary classification of breast cancer diagnosis",
        "features": cancer.feature_names.tolist(),
        "classes": ["malignant", "benign"]
    }
    
    
    # 6. Circles Dataset (Non-linear Classification)
    X_circles, y_circles = make_circles(n_samples=1000, factor=0.3, noise=0.1, random_state=42)
    X_train_circles, X_test_circles, y_train_circles, y_test_circles = train_test_split(
        X_circles, y_circles, test_size=0.2, random_state=42
    )
    
    datasets["circles"] = {
        "name": "Concentric Circles",
        "type": "classification",
        "category": "synthetic",
        "X_train": X_train_circles.tolist(),
        "X_test": X_test_circles.tolist(),
        "y_train": y_train_circles.tolist(),
        "y_test": y_test_circles.tolist(),
        "input_size": 2,
        "output_size": 2,
        "description": "Non-linearly separable concentric circles classification",
        "features": ["x_coordinate", "y_coordinate"],
        "classes": ["inner_circle", "outer_circle"]
    }
    
    # 7. Moons Dataset 
    X_moons, y_moons = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train_moons, X_test_moons, y_train_moons, y_test_moons = train_test_split(
        X_moons, y_moons, test_size=0.2, random_state=42
    )
    
    datasets["moons"] = {
        "name": "Two Moons",
        "type": "classification",
        "category": "synthetic",
        "X_train": X_train_moons.tolist(),
        "X_test": X_test_moons.tolist(),
        "y_train": y_train_moons.tolist(),
        "y_test": y_test_moons.tolist(),
        "input_size": 2,
        "output_size": 2,
        "description": "Two interleaving crescent moons classification challenge",
        "features": ["x_coordinate", "y_coordinate"],
        "classes": ["moon_1", "moon_2"]
    }
    
    # 8. Multi-class Blobs
    X_blobs, y_blobs = make_blobs(n_samples=1000, centers=4, n_features=2, 
                                  cluster_std=1.5, random_state=42)
    X_train_blobs, X_test_blobs, y_train_blobs, y_test_blobs = train_test_split(
        X_blobs, y_blobs, test_size=0.2, random_state=42
    )
    
    datasets["blobs"] = {
        "name": "Gaussian Blobs",
        "type": "classification",
        "category": "synthetic",
        "X_train": X_train_blobs.tolist(),
        "X_test": X_test_blobs.tolist(),
        "y_train": y_train_blobs.tolist(),
        "y_test": y_test_blobs.tolist(),
        "input_size": 2,
        "output_size": 4,
        "description": "Multi-class classification with Gaussian clusters",
        "features": ["feature_1", "feature_2"],
        "classes": ["cluster_1", "cluster_2", "cluster_3", "cluster_4"]
    }
    
    # === IMAGE DATASETS ===
    
    # 9. MNIST Digits (Subset for demo)
    try:
        # Create a subset of MNIST for demonstration
        np.random.seed(42)
        n_samples_per_class = 100
        n_classes = 10
        
        # Generate synthetic MNIST-like data (28x28 grayscale)
        X_mnist_demo = []
        y_mnist_demo = []
        
        for class_idx in range(n_classes):
            for _ in range(n_samples_per_class):
                # Create digit-like patterns
                img = np.random.rand(28, 28) * 0.3
                # Add some structure based on class
                if class_idx == 0:  # 0-like pattern
                    img[8:20, 8:20] = 0.1
                    img[10:18, 10:18] = 0.8
                elif class_idx == 1:  # 1-like pattern
                    img[5:23, 12:16] = 0.8
                # Add more patterns for other digits...
                
                X_mnist_demo.append(img.flatten())
                y_mnist_demo.append(class_idx)
        
        X_mnist_demo = np.array(X_mnist_demo)
        y_mnist_demo = np.array(y_mnist_demo)
        
        X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(
            X_mnist_demo, y_mnist_demo, test_size=0.2, random_state=42
        )
        
        datasets["mnist_demo"] = {
            "name": "MNIST Digits (Demo)",
            "type": "classification",
            "category": "image",
            "X_train": X_train_mnist.tolist(),
            "X_test": X_test_mnist.tolist(),
            "y_train": y_train_mnist.tolist(),
            "y_test": y_test_mnist.tolist(),
            "input_size": 784,  # 28x28 flattened
            "output_size": 10,
            "description": "Handwritten digit recognition (0-9) - Demo (Synthetic)",
            "image_shape": [28, 28, 1],
            "classes": [str(i) for i in range(10)]
        }
    except Exception as e:
        pass
    
    # TEXT/SENTIMENT DATASETS 
    
    # 10. Simple Sentiment Analysis Dataset
    # Create a simple bag-of-words sentiment dataset
    positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", 
                     "awesome", "perfect", "brilliant", "outstanding"]
    negative_words = ["bad", "terrible", "awful", "horrible", "disappointing", "poor", 
                     "worst", "hate", "disgusting", "pathetic"]
    neutral_words = ["okay", "fine", "average", "normal", "standard", "regular", 
                    "typical", "common", "usual", "ordinary"]
    
    vocab = positive_words + negative_words + neutral_words + ["movie", "film", "plot", "acting", "director"]
    vocab_size = len(vocab)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Generate synthetic sentiment data
    X_sentiment = []
    y_sentiment = []
    
    np.random.seed(42)
    for sentiment in [0, 1, 2]:  # negative, neutral, positive
        for _ in range(200):
            # Create bag-of-words vector
            bow = np.zeros(vocab_size)
            
            if sentiment == 0:  # negative
                words = np.random.choice(negative_words, size=np.random.randint(2, 6))
            elif sentiment == 1:  # neutral
                words = np.random.choice(neutral_words, size=np.random.randint(2, 6))
            else:  # positive
                words = np.random.choice(positive_words, size=np.random.randint(2, 6))
            
            # Add some common words
            words = np.concatenate([words, np.random.choice(["movie", "film"], size=1)])
            
            for word in words:
                if word in word_to_idx:
                    bow[word_to_idx[word]] += 1
            
            X_sentiment.append(bow)
            y_sentiment.append(sentiment)
    
    X_sentiment = np.array(X_sentiment)
    y_sentiment = np.array(y_sentiment)
    
    X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
        X_sentiment, y_sentiment, test_size=0.2, random_state=42
    )
    
    datasets["sentiment_analysis"] = {
        "name": "Movie Review Sentiment",
        "type": "classification",
        "category": "text",
        "X_train": X_train_sent.tolist(),
        "X_test": X_test_sent.tolist(),
        "y_train": y_train_sent.tolist(),
        "y_test": y_test_sent.tolist(),
        "input_size": vocab_size,
        "output_size": 3,
        "description": "Sentiment analysis of movie reviews (bag-of-words) - (Synthetic)",
        "features": vocab,
        "classes": ["negative", "neutral", "positive"],
        "vocab_size": vocab_size
    }
    
    
    # Enhanced Regression dataset with proper scaling
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=10.0, random_state=42)
    
    # Normalize features to [0,1] range
    scaler_X = MinMaxScaler()
    X_reg_scaled = scaler_X.fit_transform(X_reg)
    
    # Scale targets to reasonable range [0, 100] to simulate realistic regression
    scaler_y = MinMaxScaler(feature_range=(0, 100))
    y_reg_scaled = scaler_y.fit_transform(y_reg.reshape(-1, 1)).ravel()
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg_scaled, y_reg_scaled, test_size=0.2, random_state=42
    )
    
    datasets["sample_regression"] = {
        "name": "Multi-feature Regression (Normalized)",
        "type": "regression",
        "category": "synthetic",
        "X_train": X_train_reg.tolist(),
        "X_test": X_test_reg.tolist(),
        "y_train": y_train_reg.tolist(),
        "y_test": y_test_reg.tolist(),
        "input_size": X_reg_scaled.shape[1],
        "output_size": 1,
        "description": "Multi-dimensional linear regression with normalized features and realistic target range - (Synthetic)",
        "scaling_applied": True,
        "target_range": [0, 100]
    }
    
    # Enhanced Classification dataset
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                                      n_informative=10, n_redundant=10, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    datasets["sample_classification"] = {
        "name": "Multi-class Classification",
        "type": "classification",
        "category": "synthetic",
        "X_train": X_train_clf.tolist(),
        "X_test": X_test_clf.tolist(),
        "y_train": y_train_clf.tolist(),
        "y_test": y_test_clf.tolist(),
        "input_size": X_clf.shape[1],
        "output_size": 3,
        "description": "Multi-class classification with informative and redundant features"
    }
    
    

    
    # Count datasets by category and type
    dataset_summary = {}
    type_summary = {"classification": 0, "regression": 0}
    category_summary = {}
    
    for name, dataset in datasets.items():
        dataset_type = dataset["type"]
        dataset_category = dataset.get("category", "unknown")
        
        type_summary[dataset_type] += 1
        category_summary[dataset_category] = category_summary.get(dataset_category, 0) + 1
        
        # Log key datasets
        if "real" in dataset.get("data_source", "").lower() or dataset_category in ["real_estate", "medical", "image"]:
            pass
    





initialize_datasets()

@app.get("/")
async def root():
    return {"message": "NeuraViz Backend API", "version": "1.0.0", "datasets": len(datasets)}

@app.get("/datasets")
async def get_datasets():
    """Get list of available datasets with categorization"""
    dataset_info = {}
    for key, dataset in datasets.items():
        dataset_info[key] = {
            "name": dataset["name"],
            "type": dataset["type"],
            "category": dataset.get("category", "general"),
            "input_size": dataset["input_size"],
            "output_size": dataset["output_size"],
            "description": dataset.get("description", ""),
            "sample_count": (
                len(dataset["X"]) if "X" in dataset 
                else len(dataset["X_train"]) + len(dataset["X_test"])
            ),
            "features": dataset.get("features", []),
            "classes": dataset.get("classes", [])
        }
    return dataset_info

@app.get("/datasets/categories")
async def get_dataset_categories():
    """Get datasets organized by category"""
    categories = {}
    for key, dataset in datasets.items():
        category = dataset.get("category", "general")
        if category not in categories:
            categories[category] = []
        
        categories[category].append({
            "id": key,
            "name": dataset["name"],
            "type": dataset["type"],
            "description": dataset.get("description", ""),
            "input_size": dataset["input_size"],
            "output_size": dataset["output_size"]
        })
    
    return categories

@app.get("/datasets/{dataset_name}")
async def get_dataset(dataset_name: str):
    """Get specific dataset details"""
    if dataset_name not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return sanitize_recursively(datasets[dataset_name])

@app.get("/datasets/{dataset_name}/preview")
async def get_dataset_preview(dataset_name: str, samples: int = 10):
    """Get a preview of the dataset with limited samples"""
    if dataset_name not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets[dataset_name]
    preview = {
        "name": dataset["name"],
        "type": dataset["type"],
        "description": dataset.get("description", ""),
        "input_size": dataset["input_size"],
        "output_size": dataset["output_size"],
        "X_train_preview": dataset["X_train"][:samples],
        "y_train_preview": dataset["y_train"][:samples],
        "features": dataset.get("features", []),
        "classes": dataset.get("classes", []),
        "total_samples": len(dataset["X_train"]) + len(dataset["X_test"])
    }
    
    return preview

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload custom dataset (CSV format)"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        dataset_id = f"custom_{uuid.uuid4().hex[:8]}"
        
        # Determine if regression or classification
        unique_values = len(np.unique(y))
        
        # Improved dataset type detection logic
        def detect_dataset_type(y_values, unique_count):
            # Check if all values are integers (typical for classification)
            all_integers = all(isinstance(val, (int, np.integer)) for val in y_values[:100])  # Check first 100 samples
            
            # Check if values are sequential or have reasonable gaps (classification pattern)
            unique_sorted = sorted(np.unique(y_values))
            if len(unique_sorted) > 1:
                max_gap = max(unique_sorted[i+1] - unique_sorted[i] for i in range(len(unique_sorted)-1))
            else:
                max_gap = 0
            
            # Classification criteria:
            # 1. All integer values AND reasonable number of classes (≤1000) AND reasonable gaps
            # 2. OR small number of unique values regardless of type (≤20)
            if all_integers and unique_count <= 1000 and max_gap <= 100:
                return "classification"
            elif unique_count <= 20:  # Conservative threshold for small classification
                return "classification"
            else:
                return "regression"
        
        dataset_type = detect_dataset_type(y, unique_values)
        output_size = 1 if dataset_type == "regression" else unique_values
        
        datasets[dataset_id] = {
            "name": f"Custom Dataset ({file.filename})",
            "type": dataset_type,
            "category": "custom",
            "X_train": X_train.tolist(),
            "X_test": X_test.tolist(),
            "y_train": y_train.tolist(),
            "y_test": y_test.tolist(),
            "input_size": X.shape[1],
            "output_size": output_size,
            "description": f"User uploaded dataset from {file.filename}",
            "features": df.columns[:-1].tolist(),
            "target": df.columns[-1]
        }
        
        return {
            "dataset_id": dataset_id, 
            "message": "Dataset uploaded successfully",
            "type": dataset_type,
            "samples": len(X),
            "features": X.shape[1]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing dataset: {str(e)}")

@app.post("/create-network")
async def create_network(config: NetworkConfig):
    """Create a neural network from layer configuration"""
    try:
        # Use user-provided name as model ID (sanitized)
        base_model_id = sanitize_model_name(config.name or "My_Neural_Network")
        
        # Ensure uniqueness by checking if model already exists
        model_id = base_model_id
        counter = 1
        while model_id in active_models:
            model_id = f"{base_model_id}_{counter}"
            counter += 1
        

        
        # Enhanced model storage with full configuration
        active_models[model_id] = {
            "config": config.model_dump(),
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "training_history": [],
            "architecture_summary": generate_architecture_summary(config.layers),
            "parameter_count": estimate_total_parameters(config.layers),
            "display_name": config.name or "My Neural Network"  # Store original name for display
        }
        
        return {"model_id": model_id, "message": "Network created successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating network: {str(e)}")

@app.post("/train/{model_id}")
async def train_model(model_id: str, request: TrainingRequest):
    """Train a neural network model with actual PyTorch training"""


    
    if model_id not in active_models:

        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        active_models[model_id]["status"] = "training"

        
        # Get dataset
        dataset_name = request.dataset_name

        
        if not dataset_name or dataset_name not in datasets:

            raise HTTPException(status_code=400, detail="Dataset not found")
        
        dataset = datasets[dataset_name]
        dataset_type = dataset["type"]

        
        # Handle both old (X_train, X_test) and new (X, y) dataset formats
        if "X" in dataset and "y" in dataset:
            # New format: full dataset, user-controlled splits
            X = np.array(dataset["X"], dtype=np.float32)
            y = np.array(dataset["y"])
            


            
            # Perform train/validation/test split
            if request.use_test_set and request.test_split > 0:
                # Three-way split: train/validation/test
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=request.test_split, random_state=request.random_seed
                )
                # Split remaining data into train/validation
                val_ratio = request.validation_split / (request.train_split + request.validation_split)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_ratio, random_state=request.random_seed
                )

            else:
                # Two-way split: train/validation only
                val_ratio = request.validation_split / (request.train_split + request.validation_split)
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=val_ratio, random_state=request.random_seed
                )
                X_test, y_test = None, None

            
            # Store split information for later use
            split_info = {
                "train_split": request.train_split,
                "validation_split": request.validation_split, 
                "test_split": request.test_split if request.use_test_set else 0,
                "use_test_set": request.use_test_set,
                "random_seed": request.random_seed,
                "train_size": len(X_train),
                "validation_size": len(X_val),
                "test_size": len(X_test) if X_test is not None else 0
            }
            
        else:
            # Old format: predefined splits (for backwards compatibility)
            X_train = np.array(dataset["X_train"], dtype=np.float32)
            y_train = np.array(dataset["y_train"])
            X_val = np.array(dataset["X_test"], dtype=np.float32)  # Use X_test as validation
            y_val = np.array(dataset["y_test"])
            X_test, y_test = None, None  # No test set in old format
            
            split_info = {
                "train_split": request.train_split, 
                "validation_split": request.validation_split, 
                "test_split": request.test_split,
                "use_test_set": request.use_test_set, 
                "random_seed": request.random_seed,
                "train_size": len(X_train), "validation_size": len(X_val), "test_size": 0
            }

        
        # Use X_val for validation (was previously called X_test)
        X_test = X_val
        y_test = y_val
        


        
        # Check if model has convolutional layers and reshape data accordingly
        model_config = active_models[model_id]["config"]
        layers = [LayerConfig(**layer) for layer in model_config["layers"]]
        has_conv_layers = any(layer.layer_type == "conv2d" for layer in layers)
        

        for i, layer in enumerate(layers):
            pass
        
        if has_conv_layers:

            # Reshape data for CNN models
            if dataset_name in ["mnist_demo", "cifar10_demo"]:
                if dataset_name == "mnist_demo":
                    # MNIST: reshape from (batch, 784) to (batch, 1, 28, 28)
                    X_train = X_train.reshape(-1, 1, 28, 28)
                    X_test = X_test.reshape(-1, 1, 28, 28)

                elif dataset_name == "cifar10_demo":
                    # CIFAR-10: reshape from (batch, 3072) to (batch, 3, 32, 32)
                    X_train = X_train.reshape(-1, 3, 32, 32)
                    X_test = X_test.reshape(-1, 3, 32, 32)

            else:
                # For other datasets, try to infer dimensions
                feature_count = X_train.shape[1]

                if feature_count == 784:  # 28x28 grayscale
                    X_train = X_train.reshape(-1, 1, 28, 28)
                    X_test = X_test.reshape(-1, 1, 28, 28)

                elif feature_count == 3072:  # 32x32 RGB
                    X_train = X_train.reshape(-1, 3, 32, 32)
                    X_test = X_test.reshape(-1, 3, 32, 32)

                else:
                    raise Exception(f"Cannot automatically reshape data with {feature_count} features for CNN. Please use tabular data for non-image datasets.")
        
        # Additional data preprocessing for regression tasks
        # Check if dataset was already preprocessed using the new preprocessing feature
        dataset_is_preprocessed = dataset.get("preprocessing") is not None
        
        # Initialize scaler as None
        scaler = None
        
        if dataset_type == "regression" and not has_conv_layers and not dataset_is_preprocessed:

            # Apply standardization to regression features for better training stability
            
            # Only apply if data isn't already scaled (check if not in [0,1] range approximately)
            if not (np.all(X_train >= -0.1) and np.all(X_train <= 1.1)):

                scaler = StandardScaler()
                X_train_before = X_train.copy()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            else:
                pass

        elif dataset_is_preprocessed:
            pass
        

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        
        if dataset_type == "classification":
            y_train_tensor = torch.LongTensor(y_train)
            y_test_tensor = torch.LongTensor(y_test)

        else:
            y_train_tensor = torch.FloatTensor(y_train)
            y_test_tensor = torch.FloatTensor(y_test)

        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=request.network_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=request.network_config.batch_size, shuffle=False)
        


        
        # Build model
        input_size = dataset["input_size"]
        output_size = dataset["output_size"]
        

        model = build_pytorch_model(layers, input_size, output_size, dataset_type, dataset_name)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        
        # Train model

        training_history, trained_model = train_pytorch_model(
            model, train_loader, val_loader, request.network_config, dataset_type
        )

        
        # Store trained model and results
        active_models[model_id]["status"] = "trained"
        active_models[model_id]["training_history"] = training_history
        active_models[model_id]["model"] = trained_model
        active_models[model_id]["dataset_type"] = dataset_type
        active_models[model_id]["dataset_name"] = dataset_name
        active_models[model_id]["split_info"] = split_info
        
        # Store test set if available 
        has_new_format = "X" in dataset and "y" in dataset
        has_old_format = "X_train" in dataset and "X_test" in dataset and "y_train" in dataset and "y_test" in dataset
        

        
        if split_info["use_test_set"] and request.test_split > 0:
            if has_new_format:
                # New format: reconstruct the full dataset and re-split
                X_full = np.array(dataset["X"], dtype=np.float32)
                y_full = np.array(dataset["y"])
                
                _, X_test_stored, _, y_test_stored = train_test_split(
                    X_full, y_full, test_size=request.test_split, random_state=request.random_seed
                )
                
                # Apply same preprocessing that was applied to training data
                if dataset_type == "regression" and not has_conv_layers and not dataset_is_preprocessed and scaler is not None:
                    if not (np.all(X_full >= -0.1) and np.all(X_full <= 1.1)):
                        # Apply same scaler that was used for training
                        X_test_stored = scaler.transform(X_test_stored)
                
                active_models[model_id]["test_set"] = {
                    "X_test": X_test_stored.tolist(),
                    "y_test": y_test_stored.tolist(),
                    "size": len(X_test_stored)
                }

                
            elif has_old_format:
                # Old format: reconstruct full dataset and re-split  
                X_train_old = np.array(dataset["X_train"], dtype=np.float32)
                X_test_old = np.array(dataset["X_test"], dtype=np.float32)
                y_train_old = np.array(dataset["y_train"])
                y_test_old = np.array(dataset["y_test"])
                
                # Combine old train/test splits back into full dataset
                X_full = np.concatenate([X_train_old, X_test_old], axis=0)
                y_full = np.concatenate([y_train_old, y_test_old], axis=0)
                
                _, X_test_stored, _, y_test_stored = train_test_split(
                    X_full, y_full, test_size=request.test_split, random_state=request.random_seed
                )
                
                # Apply same preprocessing that was applied to training data
                if dataset_type == "regression" and not has_conv_layers and not dataset_is_preprocessed and scaler is not None:
                    # Check the combined dataset for scaling needs
                    if not (np.all(X_full >= -0.1) and np.all(X_full <= 1.1)):
                        # Apply same scaler that was used for training
                        X_test_stored = scaler.transform(X_test_stored)
                
                active_models[model_id]["test_set"] = {
                    "X_test": X_test_stored.tolist(),
                    "y_test": y_test_stored.tolist(),
                    "size": len(X_test_stored)
                }

                
            else:

                active_models[model_id]["test_set"] = None
        else:
            active_models[model_id]["test_set"] = None
        

        # Calculate final metrics based on task type
        final_metrics = {
            "final_loss": training_history[-1]["loss"],
            "best_loss": min(h["loss"] for h in training_history),
            "total_epochs": len(training_history),
            "total_parameters": sum(p.numel() for p in trained_model.parameters()),
            "trainable_parameters": sum(p.numel() for p in trained_model.parameters() if p.requires_grad)
        }
        
        # Add task-specific metrics
        if dataset_type == "classification":
            final_metrics["final_accuracy"] = training_history[-1]["accuracy"]
            final_metrics["best_accuracy"] = max(h["accuracy"] for h in training_history)

        else:  # regression
            final_metrics["final_r2_score"] = training_history[-1]["r2_score"]
            final_metrics["best_r2_score"] = max(h["r2_score"] for h in training_history)

        
        active_models[model_id]["final_metrics"] = final_metrics
        


        
        # Sanitize response data before returning
        response_data = {
            "message": "Training completed successfully", 
            "training_history": training_history,
            "final_metrics": final_metrics
        }
        
        return sanitize_recursively(response_data)
        
    except Exception as e:
        active_models[model_id]["status"] = "error"
        active_models[model_id]["error_message"] = str(e)
        raise HTTPException(status_code=400, detail=f"Error training model: {str(e)}")

@app.get("/models")
async def get_models():
    """Get list of active models"""
    return serialize_all_models(active_models)

@app.post("/evaluate/{model_id}")
async def evaluate_model(model_id: str, dataset_name: str):
    """Evaluate a trained model on test data"""
    if model_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = active_models[model_id]
    if model_data["status"] != "trained":
        raise HTTPException(status_code=400, detail="Model is not trained")
    
    if dataset_name not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        dataset = datasets[dataset_name]
        model = model_data["model"]
        dataset_type = model_data["dataset_type"]
        
        # Prepare test data
        X_test = np.array(dataset["X_test"], dtype=np.float32)
        y_test = np.array(dataset["y_test"])
        
        # Check if model has convolutional layers and reshape data accordingly
        model_config = model_data["config"]
        layers = [LayerConfig(**layer) for layer in model_config["layers"]]
        has_conv_layers = any(layer.layer_type == "conv2d" for layer in layers)
        
        if has_conv_layers:
            # Reshape data for CNN models
            if dataset_name in ["mnist_demo", "cifar10_demo"]:
                if dataset_name == "mnist_demo":
                    # MNIST: reshape from (batch, 784) to (batch, 1, 28, 28)
                    X_test = X_test.reshape(-1, 1, 28, 28)
                elif dataset_name == "cifar10_demo":
                    # CIFAR-10: reshape from (batch, 3072) to (batch, 3, 32, 32)
                    X_test = X_test.reshape(-1, 3, 32, 32)
            else:
                # For other datasets, try to infer dimensions
                feature_count = X_test.shape[1]
                if feature_count == 784:  # 28x28 grayscale
                    X_test = X_test.reshape(-1, 1, 28, 28)
                elif feature_count == 3072:  # 32x32 RGB
                    X_test = X_test.reshape(-1, 3, 32, 32)
                else:
                    raise Exception(f"Cannot automatically reshape data with {feature_count} features for CNN. Please use tabular data for non-image datasets.")
        
        X_test = torch.FloatTensor(X_test)
        
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            X_test = X_test.to(device)
            predictions = model(X_test).cpu().numpy()
        
        # Calculate metrics
        if dataset_type == "classification":
            predicted_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(predicted_classes == y_test)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, predicted_classes)
            
            return {
                "accuracy": float(accuracy),
                "predictions": predicted_classes.tolist(),
                "probabilities": predictions.tolist(),
                "confusion_matrix": cm.tolist(),
                "actual": y_test.tolist()
            }
        else:  # regression
            mse = np.mean((predictions.squeeze() - y_test) ** 2)
            mae = np.mean(np.abs(predictions.squeeze() - y_test))
            r2 = 1 - (np.sum((y_test - predictions.squeeze()) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            
            return {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2),
                "predictions": predictions.squeeze().tolist(),
                "actual": y_test.tolist()
            }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error evaluating model: {str(e)}")

@app.post("/evaluate-test-set/{model_id}")
async def evaluate_on_test_set(model_id: str):
    """Evaluate a trained model on its held-out test set"""
    if model_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = active_models[model_id]
    if model_data["status"] != "trained":
        raise HTTPException(status_code=400, detail="Model is not trained")
    
    if "test_set" not in model_data or model_data["test_set"] is None:
        raise HTTPException(status_code=400, detail="No test set available for this model")
    
    try:
        test_set = model_data["test_set"]
        model = model_data["model"]
        dataset_type = model_data["dataset_type"]
        
        # Prepare test data
        X_test = np.array(test_set["X_test"], dtype=np.float32)
        y_test = np.array(test_set["y_test"])
        

        
        # Handle CNN data reshaping if needed
        model_config = model_data["config"]
        layers = [LayerConfig(**layer) for layer in model_config["layers"]]
        has_conv_layers = any(layer.layer_type == "conv2d" for layer in layers)
        
        if has_conv_layers:
            # Reshape based on the original dataset type
            dataset_name = model_data.get("dataset_name", "")
            if dataset_name in ["mnist_demo", "cifar10_demo"]:
                if dataset_name == "mnist_demo":
                    X_test = X_test.reshape(-1, 1, 28, 28)
                elif dataset_name == "cifar10_demo":
                    X_test = X_test.reshape(-1, 3, 32, 32)
            else:
                feature_count = X_test.shape[1] if len(X_test.shape) == 2 else X_test.shape[1] * X_test.shape[2] * X_test.shape[3]
                if feature_count == 784:
                    X_test = X_test.reshape(-1, 1, 28, 28)
                elif feature_count == 3072:
                    X_test = X_test.reshape(-1, 3, 32, 32)
        
        X_test = torch.FloatTensor(X_test)
        
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            X_test = X_test.to(device)
            predictions = model(X_test).cpu().numpy()
        
        # Calculate metrics
        if dataset_type == "classification":
            predicted_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(predicted_classes == y_test)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, predicted_classes)
            
            results = {
                "test_set_size": len(y_test),
                "accuracy": float(accuracy),
                "predictions": predicted_classes.tolist(),
                "probabilities": predictions.tolist(),
                "confusion_matrix": cm.tolist(),
                "actual": y_test.tolist(),
                "dataset_name": model_data.get("dataset_name", "Unknown"),
                "split_info": model_data.get("split_info", {})
            }
            

            
        else:  # regression
            predictions_flat = predictions.squeeze()
            mse = np.mean((predictions_flat - y_test) ** 2)
            mae = np.mean(np.abs(predictions_flat - y_test))
            
            # Handle R² calculation safely
            ss_res = np.sum((y_test - predictions_flat) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            
            results = {
                "test_set_size": len(y_test),
                "mse": float(mse),
                "mae": float(mae),
                "r2_score": float(r2),
                "predictions": predictions_flat.tolist(),
                "actual": y_test.tolist(),
                "dataset_name": model_data.get("dataset_name", "Unknown"),
                "split_info": model_data.get("split_info", {})
            }
            

        
        return sanitize_recursively(results)
        
    except Exception as e:
        pass
        raise HTTPException(status_code=400, detail=f"Error evaluating model on test set: {str(e)}")

@app.post("/predict/{model_id}")
async def predict(model_id: str, input_data: List[List[float]]):
    """Make predictions using a trained model"""
    if model_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = active_models[model_id]
    if model_data["status"] != "trained":
        raise HTTPException(status_code=400, detail="Model is not trained")
    
    try:
        model = model_data["model"]
        dataset_type = model_data["dataset_type"]
        
        # Prepare input data  
        X = np.array(input_data, dtype=np.float32)
        
        # Check if model has convolutional layers and reshape data accordingly
        model_config = model_data["config"]
        layers = [LayerConfig(**layer) for layer in model_config["layers"]]
        has_conv_layers = any(layer.layer_type == "conv2d" for layer in layers)
        
        if has_conv_layers:
            # Reshape data for CNN models based on input size
            feature_count = X.shape[1] if len(X.shape) > 1 else X.shape[0]
            if feature_count == 784:  # 28x28 grayscale
                X = X.reshape(-1, 1, 28, 28)
            elif feature_count == 3072:  # 32x32 RGB
                X = X.reshape(-1, 3, 32, 32)
        
        X = torch.FloatTensor(X)
        
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            X = X.to(device)
            predictions = model(X).cpu().numpy()
        
        if dataset_type == "classification":
            predicted_classes = np.argmax(predictions, axis=1)
            return {
                "predictions": predicted_classes.tolist(),
                "probabilities": predictions.tolist()
            }
        else:
            return {
                "predictions": predictions.squeeze().tolist()
            }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making predictions: {str(e)}")

@app.get("/download/{model_id}")
async def download_model(model_id: str):
    """Download a trained model"""
    if model_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = active_models[model_id]
    if model_data["status"] != "trained":
        raise HTTPException(status_code=400, detail="Model is not trained")
    
    try:
        model = model_data["model"]
        
        # Save model to bytes
        buffer = io.BytesIO()
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_data["config"],
            'architecture_summary': model_data["architecture_summary"],
            'final_metrics': model_data["final_metrics"],
            'training_history': model_data["training_history"]
        }, buffer)
        
        # Encode as base64
        buffer.seek(0)
        model_bytes = buffer.read()
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        
        return {
            "model_data": model_b64,
            "filename": f"neuraviz_model_{model_id}.pth",
            "size": len(model_bytes)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading model: {str(e)}")

@app.get("/visualizations/{model_id}")
async def get_model_visualizations(model_id: str):
    """Generate and return model visualizations"""
    if model_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = active_models[model_id]
    if model_data["status"] != "trained":
        raise HTTPException(status_code=400, detail="Model is not trained")
    
    try:
        model = model_data["model"]
        training_history = model_data["training_history"]
        
        visualizations = {}
        
        # 1. Weight histogram
        all_weights = []
        for param in model.parameters():
            if len(param.shape) > 1:  # Only weight matrices, not biases
                all_weights.extend(param.detach().cpu().numpy().flatten())
        
        if all_weights:
            plt.figure(figsize=(10, 6))
            plt.hist(all_weights, bins=50, alpha=0.7, edgecolor='black')
            plt.title('Weight Distribution')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            weight_hist_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            visualizations['weight_histogram'] = weight_hist_b64
        
        # 2. Training curves
        if training_history:
            epochs = [h['epoch'] for h in training_history]
            # Handle null values in training history (sanitized from NaN)
            train_loss = [h.get('loss', 0) if h.get('loss') is not None else 0 for h in training_history]
            val_loss = [h.get('val_loss', 0) if h.get('val_loss') is not None else 0 for h in training_history]
            
            # Check if this is classification or regression model
            dataset_type = model_data.get("dataset_type", "classification")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss curves
            ax1.plot(epochs, train_loss, label='Training Loss', color='blue')
            ax1.plot(epochs, val_loss, label='Validation Loss', color='red')
            ax1.set_title('Training & Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Second plot: Accuracy for classification, R² for regression
            if dataset_type == "classification":
                train_metric = [h.get('accuracy', 0) if h.get('accuracy') is not None else 0 for h in training_history]
                val_metric = [h.get('val_accuracy', 0) if h.get('val_accuracy') is not None else 0 for h in training_history]
                ax2.set_title('Training & Validation Accuracy')
                ax2.set_ylabel('Accuracy')
                metric_label = 'Accuracy'
            else:  # regression
                train_metric = [h.get('r2_score', 0) if h.get('r2_score') is not None else 0 for h in training_history]
                val_metric = [h.get('val_r2_score', 0) if h.get('val_r2_score') is not None else 0 for h in training_history]
                ax2.set_title('Training & Validation R² Score')
                ax2.set_ylabel('R² Score')
                metric_label = 'R²'
            
            ax2.plot(epochs, train_metric, label=f'Training {metric_label}', color='blue')
            ax2.plot(epochs, val_metric, label=f'Validation {metric_label}', color='red')
            ax2.set_xlabel('Epoch')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            training_curves_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            visualizations['training_curves'] = training_curves_b64
        
        # 3. Model architecture visualization (text-based for now)
        architecture_text = []
        for i, layer in enumerate(model.layers):
            layer_info = f"Layer {i+1}: {type(layer).__name__}"
            if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                layer_info += f" ({layer.in_features} → {layer.out_features})"
            elif hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                layer_info += f" ({layer.in_channels} → {layer.out_channels})"
            architecture_text.append(layer_info)
        
        visualizations['architecture_text'] = architecture_text
        
        return visualizations
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating visualizations: {str(e)}")

@app.get("/models/{model_id}/status")
async def get_model_status(model_id: str):
    """Get model training status and history"""
    if model_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return serialize_model_data(active_models[model_id])

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get full model configuration and details"""
    if model_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return serialize_model_data(active_models[model_id])

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model"""
    if model_id not in active_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del active_models[model_id]
    return {"message": "Model deleted successfully"}

# PyTorch Model Building
class DynamicNeuralNetwork(nn.Module):
    """Dynamic PyTorch model built from layer configurations"""
    
    def __init__(self, layers: List[LayerConfig], input_size: int, output_size: int):
        super(DynamicNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_configs = layers
        self.input_size = input_size
        self.output_size = output_size
        
        current_size = input_size
        current_channels = None
        current_height = None
        current_width = None
        
        for i, layer_config in enumerate(layers):
            layer_type = layer_config.layer_type
            params = layer_config.params
            
            if layer_type == "dense":
                units = int(params.get("units", 64))
                
                # If coming from conv layers, flatten first
                if current_channels is not None:
                    self.layers.append(nn.Flatten())
                    current_size = current_channels * current_height * current_width
                    current_channels = None
                
                layer = nn.Linear(current_size, units)
                
                # Initialize weights
                weight_init = layer_config.weight_init or "xavier_uniform"
                self._initialize_weights(layer, weight_init)
                
                self.layers.append(layer)
                current_size = units
                
                # Add batch normalization if specified
                if layer_config.batch_norm:
                    self.layers.append(nn.BatchNorm1d(units))
                
                # Add activation (skip if 'none' or empty)
                if layer_config.activation and layer_config.activation != 'none':
                    self.layers.append(self._get_activation(layer_config.activation))
                
                # Add dropout
                if layer_config.dropout_rate and layer_config.dropout_rate > 0:
                    self.layers.append(nn.Dropout(layer_config.dropout_rate))
            
            elif layer_type == "conv2d":
                filters = int(params.get("filters", 32))
                kernel_size = int(params.get("kernel_size", 3))
                stride = int(params.get("stride", 1))
                
                # Handle padding parameter - support both int and string values
                padding_param = params.get("padding", 1)
                if isinstance(padding_param, str):
                    if padding_param.lower() == "same":
                        # 'same' padding: calculate to keep output size same as input
                        padding = (kernel_size - 1) // 2
                    elif padding_param.lower() == "valid":
                        # 'valid' padding: no padding
                        padding = 0
                    else:
                        # Try to convert string number to int
                        try:
                            padding = int(padding_param)
                        except ValueError:
                            padding = 1  # Default fallback
                else:
                    padding = int(padding_param)
                
                if i == 0:  # First layer
                    in_channels = int(params.get("input_channels", 3))
                    current_height = int(params.get("input_height", 32))
                    current_width = int(params.get("input_width", 32))
                else:
                    in_channels = current_channels
                
                layer = nn.Conv2d(in_channels, filters, kernel_size, stride, padding)
                
                # Initialize weights
                weight_init = layer_config.weight_init or "he_uniform"
                self._initialize_weights(layer, weight_init)
                
                self.layers.append(layer)
                current_channels = filters
                
                # Update spatial dimensions
                current_height = (current_height + 2 * padding - kernel_size) // stride + 1
                current_width = (current_width + 2 * padding - kernel_size) // stride + 1
                
                # Add batch normalization if specified
                if layer_config.batch_norm:
                    self.layers.append(nn.BatchNorm2d(filters))
                
                # Add activation (skip if 'none' or empty)
                if layer_config.activation and layer_config.activation != 'none':
                    self.layers.append(self._get_activation(layer_config.activation))
                
                # Add dropout
                if layer_config.dropout_rate and layer_config.dropout_rate > 0:
                    self.layers.append(nn.Dropout2d(layer_config.dropout_rate))
            
            elif layer_type == "maxpool2d":
                kernel_size = int(params.get("kernel_size", 2))
                stride = int(params.get("stride", 2))
                self.layers.append(nn.MaxPool2d(kernel_size, stride))
                
                # Update spatial dimensions
                current_height = (current_height - kernel_size) // stride + 1
                current_width = (current_width - kernel_size) // stride + 1
            
            elif layer_type == "avgpool2d":
                kernel_size = int(params.get("kernel_size", 2))
                stride = int(params.get("stride", 2))
                self.layers.append(nn.AvgPool2d(kernel_size, stride))
                
                # Update spatial dimensions
                current_height = (current_height - kernel_size) // stride + 1
                current_width = (current_width - kernel_size) // stride + 1
            
            elif layer_type == "lstm":
                units = int(params.get("units", 128))
                num_layers = int(params.get("num_layers", 1))
                bidirectional = params.get("bidirectional", False)
                
                if current_channels is not None:
                    self.layers.append(nn.Flatten())
                    current_size = current_channels * current_height * current_width
                    current_channels = None
                
                layer = nn.LSTM(current_size, units, num_layers, 
                              batch_first=True, bidirectional=bidirectional)
                self.layers.append(layer)
                
                current_size = units * (2 if bidirectional else 1)
                
                # Add dropout
                if layer_config.dropout_rate and layer_config.dropout_rate > 0:
                    self.layers.append(nn.Dropout(layer_config.dropout_rate))
            
            elif layer_type == "flatten":
                if current_channels is not None:
                    self.layers.append(nn.Flatten())
                    current_size = current_channels * current_height * current_width
                    current_channels = None
        
        # Add final output layer if needed
        if current_channels is not None:
            self.layers.append(nn.Flatten())
            current_size = current_channels * current_height * current_width
        
        if current_size != output_size:
            self.layers.append(nn.Linear(current_size, output_size))
    
    def _get_activation(self, activation: str):
        """Get activation function by name"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "swish": nn.SiLU(),
            "gelu": nn.GELU(),
            "softmax": nn.Softmax(dim=1),
            "softplus": nn.Softplus()
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_weights(self, layer, weight_init: str):
        """Initialize layer weights"""
        if hasattr(layer, 'weight'):
            if weight_init == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight)
            elif weight_init == "xavier_normal":
                nn.init.xavier_normal_(layer.weight)
            elif weight_init == "he_uniform":
                nn.init.kaiming_uniform_(layer.weight)
            elif weight_init == "he_normal":
                nn.init.kaiming_normal_(layer.weight)
            elif weight_init == "lecun_uniform":
                nn.init.uniform_(layer.weight, -np.sqrt(1/layer.weight.size(1)), np.sqrt(1/layer.weight.size(1)))
            elif weight_init == "lecun_normal":
                nn.init.normal_(layer.weight, 0, np.sqrt(1/layer.weight.size(1)))
            elif weight_init == "zeros":
                nn.init.zeros_(layer.weight)
            elif weight_init == "ones":
                nn.init.ones_(layer.weight)
            # Default: use PyTorch's default initialization
        
        # Initialize bias terms
        if hasattr(layer, 'bias') and layer.bias is not None:
            # Small positive bias for regression to avoid dead neurons
            if weight_init in ["he_uniform", "he_normal"]:
                nn.init.constant_(layer.bias, 0.1)  # Small positive bias for ReLU
            else:
                nn.init.zeros_(layer.bias)  # Standard zero bias
    
    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
                # Take the last output for sequence classification
                x = x[:, -1, :]
            else:
                x = layer(x)
        return x

def build_pytorch_model(layers: List[LayerConfig], input_size: int, output_size: int, dataset_type: str, dataset_name: str = None):
    """Build a PyTorch model from layer configuration"""
    try:
        # Validate architecture for task type
        if layers and dataset_type == "regression":
            last_layer = layers[-1]
            if (last_layer.layer_type == "dense" and 
                last_layer.activation and 
                last_layer.activation not in ['none', None] and 
                output_size == 1):
                pass
        
        # Update first conv layer parameters based on dataset
        if layers and layers[0].layer_type == "conv2d" and dataset_name:
            first_layer_params = layers[0].params
            if dataset_name == "mnist_demo":
                first_layer_params["input_channels"] = 1
                first_layer_params["input_height"] = 28
                first_layer_params["input_width"] = 28
            elif dataset_name == "cifar10_demo":
                first_layer_params["input_channels"] = 3
                first_layer_params["input_height"] = 32
                first_layer_params["input_width"] = 32
            elif input_size == 784:  # 28x28 grayscale (MNIST-like)
                first_layer_params["input_channels"] = 1
                first_layer_params["input_height"] = 28
                first_layer_params["input_width"] = 28
            elif input_size == 3072:  # 32x32 RGB (CIFAR-like)
                first_layer_params["input_channels"] = 3
                first_layer_params["input_height"] = 32
                first_layer_params["input_width"] = 32
        
        model = DynamicNeuralNetwork(layers, input_size, output_size)
        return model
    except Exception as e:
        raise Exception(f"Error building model: {str(e)}")

def get_loss_function(loss_name: str, dataset_type: str):
    """Get loss function by name"""

    
    if dataset_type == "classification":
        if loss_name in ["crossentropy", "sparse_crossentropy"]:
            return nn.CrossEntropyLoss()
        elif loss_name in ["bce", "binary_crossentropy"]:
            return nn.BCEWithLogitsLoss()
        else:

            return nn.CrossEntropyLoss()
    else:  # regression
        if loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "mae":
            return nn.L1Loss()
        elif loss_name in ["huber", "smooth_l1"]:
            return nn.SmoothL1Loss()
        else:

            return nn.MSELoss()

def get_optimizer(model, optimizer_name: str, learning_rate: float):
    """Get optimizer by name"""
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        return optim.Adam(model.parameters(), lr=learning_rate)

def calculate_accuracy(outputs, targets, dataset_type):
    """Calculate accuracy for classification or R² for regression"""
    if dataset_type == "classification":
        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        return correct / total
    else:  # regression
        # Calculate R² score with clamping to prevent infinite values
        ss_res = ((targets - outputs) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        
        # Prevent division by zero and extreme values
        if ss_tot == 0:
            return 0.0  # Perfect constant prediction
        
        r2 = 1 - (ss_res / ss_tot)
        
        # Clamp R² to reasonable bounds to prevent -inf values
        # R² can theoretically go to -infinity for very bad models
        # We clamp to [-10, 1] to keep values reasonable while still showing poor performance
        r2_clamped = torch.clamp(r2, min=-10.0, max=1.0)
        
        return r2_clamped.item()

def train_pytorch_model(model, train_loader, val_loader, config: NetworkConfig, dataset_type: str):
    """Train PyTorch model with real training loop"""
    # Force CPU usage for compatibility (can be changed to CUDA later if needed)
    device = torch.device("cpu")

    model.to(device)
    
    # Adaptive learning rate for regression tasks
    learning_rate = config.learning_rate
    
    
    
    criterion = get_loss_function(config.loss_function, dataset_type)
    optimizer = get_optimizer(model, config.optimizer, learning_rate)
    



    
    training_history = []
    

    for epoch in range(config.epochs):
        epoch_start = datetime.now()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            if dataset_type == "regression":
                output = output.squeeze()
                target = target.float()
            
            loss = criterion(output, target)
            
            # Add comprehensive debugging for first batch of first epoch
            if batch_idx == 0 and epoch == 0:










                if dataset_type == "classification":

                    
                    # Test a few predictions for classification
                    _, predicted = torch.max(output.data, 1)
                    sample_preds = predicted[:5].tolist()
                    sample_targets = target[:5].tolist()

                    
                    # Calculate accuracy manually
                    correct = (predicted == target).sum().item()
                    batch_accuracy = correct / target.size(0)

                else:  # regression
                    # For regression, show sample predictions and targets
                    sample_preds = output[:5].detach().cpu().numpy().tolist()
                    sample_targets = target[:5].detach().cpu().numpy().tolist()


                    
                    # Calculate mean absolute error for regression
                    mae = torch.mean(torch.abs(output - target)).item()

            loss.backward()
            
            # Check gradients for first batch
            if batch_idx == 0 and epoch == 0:
                total_grad_norm = 0
                param_count = 0
                for name, param in model.named_parameters():
                    pass
            optimizer.step()
            
            train_loss += loss.item()
            train_accuracy += calculate_accuracy(output, target, dataset_type)
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_train_accuracy = train_accuracy / train_batches
        
        # Validation 
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if dataset_type == "regression":
                    output = output.squeeze()
                    target = target.float()
                
                loss = criterion(output, target)
                val_loss += loss.item()
                val_accuracy += calculate_accuracy(output, target, dataset_type)
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else avg_train_loss
        avg_val_accuracy = val_accuracy / val_batches if val_batches > 0 else avg_train_accuracy
        
        # Store training history 
        epoch_data = {
            "epoch": epoch + 1,
            "loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "timestamp": datetime.now().isoformat()
        }
        
        # Accuracy for classification / R² for regression
        if dataset_type == "classification":
            epoch_data["accuracy"] = max(0, min(1, avg_train_accuracy))
            epoch_data["val_accuracy"] = max(0, min(1, avg_val_accuracy))
        else:  # regression
            epoch_data["r2_score"] = avg_train_accuracy  # R² score
            epoch_data["val_r2_score"] = avg_val_accuracy
        
        training_history.append(epoch_data)
        
        # Log progress for key epochs
        epoch_duration = (datetime.now() - epoch_start).total_seconds()
        if epoch == 0 or (epoch + 1) % max(1, config.epochs // 5) == 0 or epoch == config.epochs - 1:
            if dataset_type == "classification":
                pass
            else:  # regression
                pass
    
    # Training completion summary
    final_epoch = training_history[-1]
    if dataset_type == "classification":
        pass
    else:
        pass
    
    return training_history, model

# Helper functions
def generate_architecture_summary(layers: List[LayerConfig]) -> Dict[str, Any]:
    """Generate a summary of the network architecture"""
    summary = {
        "total_layers": len(layers),
        "trainable_layers": 0,
        "layer_types": {},
        "has_cnn": False,
        "has_rnn": False,
        "regularization": []
    }
    
    for layer in layers:
        layer_type = layer.layer_type
        
        # Count layer types
        if layer_type not in summary["layer_types"]:
            summary["layer_types"][layer_type] = 0
        summary["layer_types"][layer_type] += 1
        
        # Count trainable layers
        if layer_type in ["dense", "conv2d", "lstm"]:
            summary["trainable_layers"] += 1
        
        # Check for specific architectures
        if layer_type in ["conv2d", "maxpool2d", "avgpool2d"]:
            summary["has_cnn"] = True
        if layer_type in ["lstm", "gru", "rnn"]:
            summary["has_rnn"] = True
        
        # Check for regularization
        if layer.dropout_rate and layer.dropout_rate > 0:
            summary["regularization"].append("dropout")
        if layer.batch_norm:
            summary["regularization"].append("batch_norm")
    
    summary["regularization"] = list(set(summary["regularization"]))
    return summary

def estimate_total_parameters(layers: List[LayerConfig]) -> int:
    """Estimate total number of parameters in the network"""
    total_params = 0
    prev_size = 0
    
    for i, layer in enumerate(layers):
        layer_type = layer.layer_type
        params = layer.params
        
        if layer_type == "dense":
            units = int(params.get("units", 64))
            input_size = int(params.get("input_size", 784)) if i == 0 else prev_size
            layer_params = (input_size * units) + (units if layer.use_bias else 0)
            total_params += layer_params
            prev_size = units
            
        elif layer_type == "conv2d":
            filters = int(params.get("filters", 32))
            kernel_size = int(params.get("kernel_size", 3))
            input_channels = int(params.get("input_channels", 3)) if i == 0 else prev_size
            layer_params = (kernel_size * kernel_size * input_channels * filters) + (filters if layer.use_bias else 0)
            total_params += layer_params
            prev_size = filters
            
        elif layer_type == "lstm":
            units = int(params.get("units", 128))
            input_size = int(params.get("input_size", 100)) if i == 0 else prev_size
            # LSTM has 4 gates (input, forget, output, candidate)
            layer_params = 4 * (input_size * units + units * units + units)
            total_params += layer_params
            prev_size = units
    
    return total_params

@app.get("/datasets/{dataset_name}/preprocessing-info")
async def get_preprocessing_info(dataset_name: str):
    """Analyze dataset and provide preprocessing recommendations"""
    if dataset_name not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        dataset = datasets[dataset_name]
        
        try:
            if "X" in dataset and "y" in dataset:
                X_data = np.array(dataset["X"], dtype=np.float32)
                y_data = np.array(dataset["y"])
            elif "X_train" in dataset and "y_train" in dataset:
                X_data = np.array(dataset["X_train"], dtype=np.float32)
                y_data = np.array(dataset["y_train"])
            else:
                raise HTTPException(status_code=400, detail=f"Dataset {dataset_name} has invalid format - missing required data keys")
                
        except (ValueError, TypeError) as e:

            raise HTTPException(status_code=400, detail=f"Dataset contains non-numeric data that cannot be preprocessed: {str(e)}")
        
        # Analyze features
        analysis = {
            "dataset_name": dataset_name,
            "dataset_type": dataset["type"],
            "feature_analysis": [],
            "target_analysis": {},
            "recommendations": [],
            "preprocessing_needed": False
        }
        
        # Analyze each feature
        for i in range(X_data.shape[1]):
            feature = X_data[:, i]
            feature_min = float(np.min(feature)) if len(feature) > 0 else 0.0
            feature_max = float(np.max(feature)) if len(feature) > 0 else 0.0
            feature_mean = float(np.mean(feature)) if len(feature) > 0 else 0.0
            feature_std = float(np.std(feature)) if len(feature) > 0 else 0.0
            feature_range = float(np.ptp(feature)) if len(feature) > 0 else 0.0
            
            # Safe outlier detection
            if feature_std > 0 and len(feature) > 0:
                outlier_threshold = 3 * feature_std
                outliers = np.abs(feature - feature_mean) > outlier_threshold
                has_outliers = bool(np.sum(outliers) > 0)
            else:
                has_outliers = False
            
            # Normalization and standardization checks
            is_normalized = bool(np.all(feature >= -0.1) and np.all(feature <= 1.1)) if len(feature) > 0 else False
            is_standardized = bool(abs(feature_mean) < 0.1 and abs(feature_std - 1.0) < 0.1) if feature_std > 0 else False
            
            feature_info = {
                "index": i,
                "name": dataset.get("features", [f"Feature_{i}"])[i] if i < len(dataset.get("features", [])) else f"Feature_{i}",
                "min": feature_min,
                "max": feature_max,
                "mean": feature_mean,
                "std": feature_std,
                "range": feature_range,
                "has_outliers": has_outliers,
                "is_normalized": is_normalized,
                "is_standardized": is_standardized
            }
            analysis["feature_analysis"].append(feature_info)
        
        # Target analysis
        if dataset["type"] == "regression":
            # Safe target statistics
            target_min = float(np.min(y_data)) if len(y_data) > 0 else 0.0
            target_max = float(np.max(y_data)) if len(y_data) > 0 else 0.0
            target_mean = float(np.mean(y_data)) if len(y_data) > 0 else 0.0
            target_std = float(np.std(y_data)) if len(y_data) > 0 else 0.0
            target_range = float(np.ptp(y_data)) if len(y_data) > 0 else 0.0
            
            analysis["target_analysis"] = {
                "min": target_min,
                "max": target_max,
                "mean": target_mean,
                "std": target_std,
                "range": target_range
            }
        else:
            unique_classes = np.unique(y_data)
            class_counts = {int(cls): int(np.sum(y_data == cls)) for cls in unique_classes}
            
            # Safe division for class balance check
            if len(class_counts.values()) > 0 and min(class_counts.values()) > 0:
                is_balanced = max(class_counts.values()) / min(class_counts.values()) < 2.0
            else:
                is_balanced = False
            
            analysis["target_analysis"] = {
                "classes": len(unique_classes),
                "class_distribution": class_counts,
                "is_balanced": bool(is_balanced)
            }
        
        # Generate recommendations
        feature_ranges = [f["range"] for f in analysis["feature_analysis"]]
        # Check for scale differences and don't divide by zero
        if len(feature_ranges) > 1:
            if min(feature_ranges) == 0:
                # Some features have zero range 
                zero_range_count = sum(1 for r in feature_ranges if r == 0)
                analysis["recommendations"].append({
                    "type": "info",
                    "priority": "medium",
                    "message": f"{zero_range_count} features have constant values  These features provide no information for training.",
                    "suggested_method": "StandardScaler"
                })
                # Check remaining non-zero range features
                non_zero_ranges = [r for r in feature_ranges if r > 0]
                if len(non_zero_ranges) > 1 and max(non_zero_ranges) / min(non_zero_ranges) > 100:
                    analysis["preprocessing_needed"] = True
            elif max(feature_ranges) / min(feature_ranges) > 100:
                analysis["recommendations"].append({
                    "type": "scaling",
                    "priority": "high",
                    "message": "Features have very different scales. Standardization recommended.",
                    "suggested_method": "StandardScaler"
                })
                analysis["preprocessing_needed"] = True
        
        # Check if already preprocessed
        all_normalized = all(f["is_normalized"] for f in analysis["feature_analysis"])
        all_standardized = all(f["is_standardized"] for f in analysis["feature_analysis"])
        
        if all_normalized:
            analysis["recommendations"].append({
                "type": "info",
                "priority": "low",
                "message": "Data appears to be normalized (0-1 range).",
                "suggested_method": "none"
            })
        elif all_standardized:
            analysis["recommendations"].append({
                "type": "info",
                "priority": "low",
                "message": "Data appears to be standardized (mean=0, std=1).",
                "suggested_method": "none"
            })
        elif not analysis["preprocessing_needed"]:
            analysis["recommendations"].append({
                "type": "scaling",
                "priority": "medium",
                "message": "Consider standardization for neural network training.",
                "suggested_method": "StandardScaler"
            })
        
        outlier_features = [f for f in analysis["feature_analysis"] if f["has_outliers"]]
        if outlier_features:
            analysis["recommendations"].append({
                "type": "outliers",
                "priority": "medium",
                "message": f"Outliers detected in {len(outlier_features)} features. Consider RobustScaler.",
                "suggested_method": "RobustScaler"
            })
        
        return sanitize_recursively(analysis)
        
    except Exception as e:


        raise HTTPException(status_code=500, detail=f"Error analyzing dataset for preprocessing: {str(e)}")

class PreprocessingConfig(BaseModel):
    dataset_name: str
    feature_preprocessing: Dict[int, str]  
    target_preprocessing: Optional[str] = None
    remove_outliers: bool = False
    outlier_method: str = "iqr" 
    custom_ranges: Optional[Dict[int, List[float]]] = None  

@app.post("/datasets/{dataset_name}/preprocess")
async def preprocess_dataset(dataset_name: str, config: PreprocessingConfig):
    """Apply preprocessing to a dataset"""
    if dataset_name not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        dataset = datasets[dataset_name].copy()
        
        # Handle both dataset formats
        has_new_format = "X" in dataset and "y" in dataset
        has_old_format = "X_train" in dataset and "X_test" in dataset and "y_train" in dataset and "y_test" in dataset
        
        if has_new_format:
            # New format: split the full dataset
            from sklearn.model_selection import train_test_split
            X_full = np.array(dataset["X"], dtype=np.float32)
            y_full = np.array(dataset["y"])
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_full, test_size=0.3, random_state=42
            )
        elif has_old_format:
            # Old format: use existing splits
            X_train = np.array(dataset["X_train"], dtype=np.float32)
            X_test = np.array(dataset["X_test"], dtype=np.float32)
            y_train = np.array(dataset["y_train"])
            y_test = np.array(dataset["y_test"])
        else:
            raise ValueError("Dataset format not supported. Expected either (X, y) or (X_train, X_test, y_train, y_test)")
        
        preprocessing_info = {
            "applied_transformations": [],
            "feature_scalers": {},
            "target_scaler": None,
            "removed_samples": 0
        }
        
        # Remove outliers if requested
        if config.remove_outliers:
            if config.outlier_method == "iqr":
                # IQR method
                Q1 = np.percentile(X_train, 25, axis=0)
                Q3 = np.percentile(X_train, 75, axis=0)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Find outlier samples
                outlier_mask = np.any((X_train < lower_bound) | (X_train > upper_bound), axis=1)
                
            else:  # zscore method
                z_scores = np.abs((X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0))
                outlier_mask = np.any(z_scores > 3, axis=1)
            
            # Remove outliers 
            original_train_size = len(X_train)
            X_train = X_train[~outlier_mask]
            y_train = y_train[~outlier_mask]
            preprocessing_info["removed_samples"] = original_train_size - len(X_train)
            preprocessing_info["applied_transformations"].append(f"outlier_removal_{config.outlier_method}")
        
        # Apply feature preprocessing
        for feature_idx, scaler_type in config.feature_preprocessing.items():
            if feature_idx >= X_train.shape[1]:
                continue
                
            feature_train = X_train[:, feature_idx].reshape(-1, 1)
            feature_test = X_test[:, feature_idx].reshape(-1, 1)
            
            if scaler_type == "StandardScaler":
                scaler = StandardScaler()
                X_train[:, feature_idx] = scaler.fit_transform(feature_train).flatten()
                X_test[:, feature_idx] = scaler.transform(feature_test).flatten()
                preprocessing_info["feature_scalers"][feature_idx] = {
                    "type": "StandardScaler",
                    "mean": float(scaler.mean_[0]),
                    "scale": float(scaler.scale_[0])
                }
                
            elif scaler_type == "MinMaxScaler":
                scaler = MinMaxScaler()
                X_train[:, feature_idx] = scaler.fit_transform(feature_train).flatten()
                X_test[:, feature_idx] = scaler.transform(feature_test).flatten()
                preprocessing_info["feature_scalers"][feature_idx] = {
                    "type": "MinMaxScaler",
                    "min": float(scaler.data_min_[0]),
                    "max": float(scaler.data_max_[0]),
                    "scale": float(scaler.scale_[0])
                }
                
            elif scaler_type == "RobustScaler":
                scaler = RobustScaler()
                X_train[:, feature_idx] = scaler.fit_transform(feature_train).flatten()
                X_test[:, feature_idx] = scaler.transform(feature_test).flatten()
                preprocessing_info["feature_scalers"][feature_idx] = {
                    "type": "RobustScaler",
                    "center": float(scaler.center_[0]),
                    "scale": float(scaler.scale_[0])
                }
                
            elif scaler_type == "custom" and config.custom_ranges and feature_idx in config.custom_ranges:
                min_val, max_val = config.custom_ranges[feature_idx]
                original_min = np.min(feature_train)
                original_max = np.max(feature_train)
                
                # Custom scaling
                X_train[:, feature_idx] = ((feature_train.flatten() - original_min) / 
                                         (original_max - original_min) * (max_val - min_val) + min_val)
                X_test[:, feature_idx] = ((feature_test.flatten() - original_min) / 
                                        (original_max - original_min) * (max_val - min_val) + min_val)
                
                preprocessing_info["feature_scalers"][feature_idx] = {
                    "type": "custom",
                    "original_min": float(original_min),
                    "original_max": float(original_max),
                    "target_min": min_val,
                    "target_max": max_val
                }
        
        # Apply target preprocessing for regression
        if config.target_preprocessing and dataset["type"] == "regression":
            if config.target_preprocessing == "StandardScaler":
                scaler = StandardScaler()
                y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()
                preprocessing_info["target_scaler"] = {
                    "type": "StandardScaler",
                    "mean": float(scaler.mean_[0]),
                    "scale": float(scaler.scale_[0])
                }
                
            elif config.target_preprocessing == "MinMaxScaler":
                scaler = MinMaxScaler()
                y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()
                preprocessing_info["target_scaler"] = {
                    "type": "MinMaxScaler",
                    "min": float(scaler.data_min_[0]),
                    "max": float(scaler.data_max_[0]),
                    "scale": float(scaler.scale_[0])
                }
        
        # Create new dataset with preprocessing applied
        processed_dataset_id = f"{dataset_name}_processed_{uuid.uuid4().hex[:8]}"
        
        datasets[processed_dataset_id] = {
            **dataset,
            "name": f"{dataset['name']} (Processed)",
            "X_train": X_train.tolist(),
            "X_test": X_test.tolist(),
            "y_train": y_train.tolist(),
            "y_test": y_test.tolist(),
            "preprocessing": preprocessing_info,
            "original_dataset": dataset_name,
            "category": "preprocessed"
        }
        
        preprocessing_info["applied_transformations"] = list(set(preprocessing_info["applied_transformations"] + 
                                                               [f"feature_{i}_{scaler}" for i, scaler in config.feature_preprocessing.items()]))
        
        return {
            "message": "Preprocessing applied successfully",
            "processed_dataset_id": processed_dataset_id,
            "preprocessing_info": preprocessing_info,
            "new_dataset_stats": {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features": X_train.shape[1]
            }
        }
        
    except Exception as e:

        raise HTTPException(status_code=400, detail=f"Error preprocessing dataset: {str(e)}")

@app.get("/datasets/{dataset_name}/preprocessing-preview")
async def preview_preprocessing(dataset_name: str, feature_index: int, scaler_type: str):
    """Preview preprocessing effects on a specific feature"""
    if dataset_name not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if feature_index < 0 or feature_index >= datasets[dataset_name]["input_size"]:
        raise HTTPException(status_code=400, detail="Invalid feature index")
    
    try:
        dataset = datasets[dataset_name]
        
        if "X" in dataset:
            X_data = np.array(dataset["X"], dtype=np.float32)
        elif "X_train" in dataset:
            X_data = np.array(dataset["X_train"], dtype=np.float32)
        else:
            raise HTTPException(status_code=400, detail="Dataset format not supported")
            
        feature_data = X_data[:, feature_index]
        
        # Apply the requested scaling
        feature_reshaped = feature_data.reshape(-1, 1)
        
        if scaler_type == "StandardScaler":
            scaler = StandardScaler()
            scaled_feature = scaler.fit_transform(feature_reshaped).flatten()
        elif scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler()
            scaled_feature = scaler.fit_transform(feature_reshaped).flatten()
        elif scaler_type == "RobustScaler":
            scaler = RobustScaler()
            scaled_feature = scaler.fit_transform(feature_reshaped).flatten()
        else:
            scaled_feature = feature_data
        
        # Calculate statistics
        original_stats = {
            "min": float(np.min(feature_data)),
            "max": float(np.max(feature_data)),
            "mean": float(np.mean(feature_data)),
            "std": float(np.std(feature_data)),
            "q25": float(np.percentile(feature_data, 25)),
            "q75": float(np.percentile(feature_data, 75))
        }
        
        scaled_stats = {
            "min": float(np.min(scaled_feature)),
            "max": float(np.max(scaled_feature)),
            "mean": float(np.mean(scaled_feature)),
            "std": float(np.std(scaled_feature)),
            "q25": float(np.percentile(scaled_feature, 25)),
            "q75": float(np.percentile(scaled_feature, 75))
        }
        
        return {
            "feature_name": dataset.get("features", [f"Feature_{feature_index}"])[feature_index] if feature_index < len(dataset.get("features", [])) else f"Feature_{feature_index}",
            "scaler_type": scaler_type,
            "original_stats": original_stats,
            "scaled_stats": scaled_stats,
            "sample_original": feature_data[:10].tolist(),
            "sample_scaled": scaled_feature[:10].tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error previewing preprocessing: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
