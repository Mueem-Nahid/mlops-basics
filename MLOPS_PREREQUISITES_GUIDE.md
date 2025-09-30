# MLOps with Cloud - Prerequisites Guide

A comprehensive step-by-step guide with code examples to prepare for the MLOps course.

---

## Table of Contents

1. [Python Fundamentals for ML](#1-python-fundamentals-for-ml)
2. [Data Transformation Basics](#2-data-transformation-basics)
3. [Machine Learning Model Architectures](#3-machine-learning-model-architectures)
4. [Linux & Shell Basics](#4-linux--shell-basics)
5. [Git & GitHub Basics](#5-git--github-basics)
6. [AWS Fundamentals](#6-aws-fundamentals)
7. [Docker Basics](#7-docker-basics)
8. [Practice Projects](#8-practice-projects)

---

## 1. Python Fundamentals for ML

### 1.1 Essential Libraries

```python
# Install essential libraries
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow torch

# Import commonly used libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

### 1.2 Working with NumPy

```python
# Creating arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Array operations
print(arr.shape)  # (5,)
print(matrix.shape)  # (2, 3)

# Mathematical operations
arr_squared = arr ** 2
arr_mean = np.mean(arr)
arr_std = np.std(arr)

# Matrix operations
matrix_transpose = matrix.T
dot_product = np.dot(matrix, matrix.T)

# Generating random data
random_data = np.random.randn(100, 5)  # 100 samples, 5 features
```

### 1.3 Working with Pandas

```python
# Creating DataFrames
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50],
    'target': [0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

# Basic operations
print(df.head())
print(df.describe())
print(df.info())

# Filtering data
filtered_df = df[df['target'] == 1]

# Handling missing values
df_filled = df.fillna(df.mean())

# Grouping and aggregation
grouped = df.groupby('target').mean()
```

---

## 2. Data Transformation Basics

### 2.1 Data Cleaning

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# Check for missing values
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
# Method 1: Drop rows with missing values
df_clean = df.dropna()

# Method 2: Fill with mean/median/mode
df['column'] = df['column'].fillna(df['column'].mean())

# Method 3: Forward/backward fill
df_ffill = df.fillna(method='ffill')

# Handle outliers using IQR method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_no_outliers = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]
```

### 2.2 Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler (z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler (scale to range [0, 1])
min_max_scaler = MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)

# RobustScaler (robust to outliers)
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
```

### 2.3 Feature Engineering

```python
# One-Hot Encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=['category_column'])

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['encoded_column'] = le.fit_transform(df['category_column'])

# Creating new features
df['feature_ratio'] = df['feature1'] / (df['feature2'] + 1)
df['feature_interaction'] = df['feature1'] * df['feature2']

# Binning continuous variables
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], 
                         labels=['child', 'young_adult', 'adult', 'senior'])

# Log transformation for skewed data
df['log_feature'] = np.log1p(df['skewed_feature'])

# Creating polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

### 2.4 Data Splitting

```python
from sklearn.model_selection import train_test_split

# Basic train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Stratified split (for imbalanced datasets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

---

## 3. Machine Learning Model Architectures

### 3.1 Classical Machine Learning

#### Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}, R2: {r2}")
```

#### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

#### Decision Trees and Random Forest

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
```

#### Gradient Boosting (XGBoost, LightGBM)

```python
import xgboost as xgb
from lightgbm import LGBMClassifier

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# LightGBM
lgb_model = LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
lgb_model.fit(X_train, y_train)
```

### 3.2 Neural Networks (NN)

#### Simple Neural Network with TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create a simple feedforward neural network
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")
```

#### Neural Network with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Initialize model
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
model = SimpleNN(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

### 3.3 Convolutional Neural Networks (CNN)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# CNN for image classification
model = models.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val, y_val)
)
```

#### CNN with PyTorch

```python
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN()
```

### 3.4 Recurrent Neural Networks (RNN, LSTM, GRU)

#### LSTM for Sequence Data

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# LSTM model for time series / sequence data
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val)
)
```

#### LSTM with PyTorch

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize
input_size = 10
hidden_size = 50
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
```

---

## 4. Linux & Shell Basics

### 4.1 Essential Linux Commands

```bash
# Navigation
pwd                    # Print working directory
ls                     # List files
ls -la                 # List all files with details
cd /path/to/directory  # Change directory
cd ..                  # Go up one level
cd ~                   # Go to home directory

# File Operations
touch file.txt         # Create empty file
mkdir new_folder       # Create directory
mkdir -p path/to/dir   # Create nested directories
cp file.txt copy.txt   # Copy file
cp -r folder/ dest/    # Copy directory recursively
mv file.txt newname.txt # Rename/move file
rm file.txt            # Remove file
rm -rf folder/         # Remove directory recursively
cat file.txt           # Display file content
less file.txt          # View file (paginated)
head -n 10 file.txt    # First 10 lines
tail -n 10 file.txt    # Last 10 lines
tail -f logfile.log    # Follow file updates

# File Permissions
chmod +x script.sh     # Make file executable
chmod 755 script.sh    # Set specific permissions
chown user:group file  # Change ownership

# Search and Find
find . -name "*.py"    # Find files by name
grep "pattern" file.txt # Search in file
grep -r "pattern" .    # Recursive search
locate filename        # Quick file search

# Process Management
ps aux                 # List all processes
top                    # Monitor processes
htop                   # Interactive process viewer
kill PID               # Kill process by ID
killall process_name   # Kill by name
bg                     # Background process
fg                     # Foreground process

# System Information
df -h                  # Disk space
du -sh folder/         # Folder size
free -h                # Memory usage
uname -a               # System information
whoami                 # Current user
```

### 4.2 Shell Scripting Basics

```bash
#!/bin/bash

# Variables
NAME="MLOps"
echo "Hello, $NAME"

# User Input
read -p "Enter your name: " USERNAME
echo "Welcome, $USERNAME"

# Conditional Statements
if [ -f "file.txt" ]; then
    echo "File exists"
else
    echo "File does not exist"
fi

# Loops
for i in {1..5}; do
    echo "Iteration $i"
done

for file in *.py; do
    echo "Processing $file"
done

while read line; do
    echo "$line"
done < file.txt

# Functions
function greet() {
    echo "Hello, $1"
}
greet "World"

# Command substitution
CURRENT_DATE=$(date +%Y-%m-%d)
echo "Today is $CURRENT_DATE"

# Environment Variables
export MODEL_PATH="/models"
echo $MODEL_PATH
```

### 4.3 WSL2 (Windows Subsystem for Linux)

```bash
# Install WSL2 (PowerShell as Administrator)
wsl --install

# List available distributions
wsl --list --online

# Install specific distribution
wsl --install -d Ubuntu-22.04

# Set default WSL version
wsl --set-default-version 2

# Access WSL from Windows
wsl

# Access Windows files from WSL
cd /mnt/c/Users/YourUsername

# Install packages in WSL
sudo apt update
sudo apt install python3-pip
sudo apt install docker.io
```

---

## 5. Git & GitHub Basics

### 5.1 Git Configuration

```bash
# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Check configuration
git config --list

# Set default editor
git config --global core.editor "code --wait"
```

### 5.2 Basic Git Workflow

```bash
# Initialize repository
git init

# Clone repository
git clone https://github.com/username/repo.git

# Check status
git status

# Stage changes
git add file.txt           # Add specific file
git add .                  # Add all changes
git add *.py               # Add all Python files

# Commit changes
git commit -m "Descriptive commit message"

# Push changes
git push origin main

# Pull changes
git pull origin main

# View commit history
git log
git log --oneline --graph --all
```

### 5.3 Branching

```bash
# Create branch
git branch feature-branch

# Switch to branch
git checkout feature-branch

# Create and switch (shorthand)
git checkout -b feature-branch

# List branches
git branch
git branch -a  # Include remote branches

# Merge branch
git checkout main
git merge feature-branch

# Delete branch
git branch -d feature-branch
git push origin --delete feature-branch
```

### 5.4 Advanced Git Operations

```bash
# Stash changes
git stash
git stash list
git stash apply
git stash pop

# Undo changes
git checkout -- file.txt   # Discard local changes
git reset HEAD file.txt    # Unstage file
git reset --hard HEAD      # Reset to last commit
git reset --hard origin/main # Reset to remote

# View differences
git diff                   # Unstaged changes
git diff --staged          # Staged changes
git diff branch1..branch2  # Compare branches

# Remote repositories
git remote -v
git remote add origin https://github.com/user/repo.git
git remote remove origin

# Tags
git tag v1.0.0
git push origin v1.0.0
```

### 5.5 Git for ML Projects

```bash
# Create .gitignore for Python/ML projects
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Data
data/
*.csv
*.h5
*.pkl
*.joblib

# Models
models/
*.pth
*.h5
checkpoints/

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
EOF

# Track large files with Git LFS
git lfs install
git lfs track "*.pth"
git lfs track "*.h5"
git add .gitattributes
```

---

## 6. AWS Fundamentals

### 6.1 AWS CLI Setup

```bash
# Install AWS CLI
pip install awscli

# Configure AWS CLI
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., us-east-1)
# - Default output format (json)

# Test configuration
aws sts get-caller-identity
```

### 6.2 S3 (Simple Storage Service) Basics

```bash
# List buckets
aws s3 ls

# Create bucket
aws s3 mb s3://my-mlops-bucket

# Upload file
aws s3 cp model.pkl s3://my-mlops-bucket/models/

# Upload directory
aws s3 sync ./data s3://my-mlops-bucket/data/

# Download file
aws s3 cp s3://my-mlops-bucket/models/model.pkl ./

# Download directory
aws s3 sync s3://my-mlops-bucket/data/ ./data/

# List bucket contents
aws s3 ls s3://my-mlops-bucket/

# Delete file
aws s3 rm s3://my-mlops-bucket/models/model.pkl

# Delete bucket
aws s3 rb s3://my-mlops-bucket --force
```

### 6.3 S3 with Python (boto3)

```python
import boto3
import os

# Create S3 client
s3 = boto3.client('s3')

# Upload file
def upload_to_s3(file_path, bucket, key):
    s3.upload_file(file_path, bucket, key)
    print(f"Uploaded {file_path} to s3://{bucket}/{key}")

# Download file
def download_from_s3(bucket, key, file_path):
    s3.download_file(bucket, key, file_path)
    print(f"Downloaded s3://{bucket}/{key} to {file_path}")

# List objects
def list_s3_objects(bucket, prefix=''):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        for obj in response['Contents']:
            print(obj['Key'])

# Example usage
upload_to_s3('model.pkl', 'my-mlops-bucket', 'models/model.pkl')
download_from_s3('my-mlops-bucket', 'models/model.pkl', 'downloaded_model.pkl')
list_s3_objects('my-mlops-bucket', 'models/')
```

### 6.4 EC2 Basics

```bash
# List EC2 instances
aws ec2 describe-instances

# Start instance
aws ec2 start-instances --instance-ids i-1234567890abcdef0

# Stop instance
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Connect to EC2 via SSH
ssh -i "key.pem" ubuntu@ec2-xx-xx-xx-xx.compute-1.amazonaws.com
```

---

## 7. Docker Basics

### 7.1 Docker Installation

```bash
# Install Docker on Ubuntu
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (no sudo needed)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker run hello-world
```

### 7.2 Basic Docker Commands

```bash
# Pull image
docker pull python:3.9

# List images
docker images

# Run container
docker run -it python:3.9 python

# Run container with name
docker run --name my-container python:3.9 python

# Run in detached mode
docker run -d nginx

# List running containers
docker ps

# List all containers
docker ps -a

# Stop container
docker stop container_id

# Start container
docker start container_id

# Remove container
docker rm container_id

# Remove image
docker rmi image_id

# View logs
docker logs container_id

# Execute command in running container
docker exec -it container_id bash
```

### 7.3 Create a Dockerfile for ML

```dockerfile
# Dockerfile for ML application
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV MODEL_PATH=/app/models

# Run application
CMD ["python", "app.py"]
```

```txt
# requirements.txt
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.2.2
fastapi==0.95.0
uvicorn==0.21.0
```

### 7.4 Build and Run Docker Image

```bash
# Build image
docker build -t my-ml-app:v1 .

# Run container
docker run -p 8000:8000 my-ml-app:v1

# Run with volume mount
docker run -v $(pwd)/data:/app/data -p 8000:8000 my-ml-app:v1

# Run with environment variables
docker run -e MODEL_PATH=/models -p 8000:8000 my-ml-app:v1
```

### 7.5 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

```bash
# Start services
docker-compose up

# Start in detached mode
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

---

## 8. Practice Projects

### Project 1: Simple ML Pipeline

```python
# complete_ml_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Data
df = pd.read_csv('data.csv')

# 2. Exploratory Data Analysis
print(df.head())
print(df.info())
print(df.describe())

# 3. Data Preprocessing
# Handle missing values
df = df.dropna()

# Feature engineering
X = df.drop('target', axis=1)
y = df['target']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Model Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# 8. Save Model
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 9. Load and Use Model
loaded_model = joblib.load('model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Make prediction
new_data = np.array([[...]])  # Your data
new_data_scaled = loaded_scaler.transform(new_data)
prediction = loaded_model.predict(new_data_scaled)
print(f"Prediction: {prediction}")
```

### Project 2: Simple REST API for ML Model

```python
# app.py - FastAPI application
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI(title="ML Model API")

# Load model and scaler at startup
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Define request/response models
class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

@app.get("/")
def read_root():
    return {"message": "ML Model API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Prepare data
        data = np.array(request.features).reshape(1, -1)
        data_scaled = scaler.transform(data)
        
        # Make prediction
        prediction = int(model.predict(data_scaled)[0])
        probability = float(model.predict_proba(data_scaled)[0][prediction])
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run with: uvicorn app:app --reload
```

### Project 3: Data Version Control with DVC

```bash
# Initialize DVC
git init
dvc init

# Add remote storage (S3)
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Track data file
dvc add data/dataset.csv

# Commit DVC files
git add data/dataset.csv.dvc data/.gitignore
git commit -m "Add dataset"

# Push data to remote
dvc push

# Pull data from remote
dvc pull

# Create a pipeline
dvc run -n preprocess \
    -d data/raw_data.csv \
    -o data/processed_data.csv \
    python preprocess.py

dvc run -n train \
    -d data/processed_data.csv \
    -d train.py \
    -o models/model.pkl \
    python train.py
```

---

## 9. Learning Path & Resources

### 9.1 Recommended Learning Order

1. **Week 1-2: Python & Data Basics**
   - Master NumPy and Pandas
   - Practice data cleaning and transformation
   - Complete 3-5 Kaggle notebooks

2. **Week 3-4: Classical ML**
   - Implement Linear Regression, Logistic Regression
   - Build Decision Trees, Random Forest
   - Practice with scikit-learn
   - Complete 2-3 end-to-end ML projects

3. **Week 5-6: Deep Learning Fundamentals**
   - Learn Neural Networks basics
   - Implement CNN for image classification
   - Implement LSTM for sequence data
   - Build 1-2 DL projects

4. **Week 7: Linux & Shell**
   - Practice Linux commands daily
   - Write shell scripts
   - Get comfortable with WSL2 (if on Windows)

5. **Week 8: Git & GitHub**
   - Create repositories
   - Practice branching and merging
   - Collaborate on open-source projects

6. **Week 9: AWS Basics**
   - Set up AWS account
   - Practice with S3, EC2
   - Use boto3 for programmatic access

7. **Week 10: Docker**
   - Install Docker
   - Create Dockerfiles
   - Build and run containers
   - Use Docker Compose

### 9.2 Practice Exercises

#### Exercise 1: Build a Classification Model
```python
# Task: Build a binary classifier for the Titanic dataset
# Steps:
# 1. Load data from Kaggle
# 2. Perform EDA
# 3. Handle missing values
# 4. Feature engineering
# 5. Train multiple models
# 6. Compare performance
# 7. Save best model
```

#### Exercise 2: Create a Dockerized ML API
```bash
# Task: Containerize your ML model API
# Steps:
# 1. Create FastAPI app
# 2. Write Dockerfile
# 3. Build Docker image
# 4. Run container
# 5. Test API endpoints
# 6. Push to Docker Hub
```

#### Exercise 3: Version Control with Git
```bash
# Task: Manage ML project with Git
# Steps:
# 1. Initialize repository
# 2. Create feature branches
# 3. Track changes
# 4. Write meaningful commits
# 5. Push to GitHub
# 6. Create pull requests
```

### 9.3 Useful Resources

#### Documentation
- **Python**: https://docs.python.org/3/
- **NumPy**: https://numpy.org/doc/
- **Pandas**: https://pandas.pydata.org/docs/
- **Scikit-learn**: https://scikit-learn.org/stable/
- **TensorFlow**: https://www.tensorflow.org/api_docs
- **PyTorch**: https://pytorch.org/docs/
- **Docker**: https://docs.docker.com/
- **Kubernetes**: https://kubernetes.io/docs/
- **AWS**: https://docs.aws.amazon.com/

#### Interactive Learning
- **Kaggle Learn**: https://www.kaggle.com/learn
- **Google Colab**: https://colab.research.google.com/
- **DataCamp**: https://www.datacamp.com/
- **Coursera**: Machine Learning specializations
- **Fast.ai**: Practical Deep Learning

#### YouTube Channels
- StatQuest with Josh Starmer
- Sentdex
- Tech With Tim
- Krish Naik
- CodeBasics

#### Books
- "Hands-On Machine Learning" by Aurélien Géron
- "Python Data Science Handbook" by Jake VanderPlas
- "Deep Learning with Python" by François Chollet
- "Designing Data-Intensive Applications" by Martin Kleppmann

---

## 10. Checklist: Are You Ready?

Before starting the MLOps course, ensure you can:

### Python & ML
- [ ] Write Python scripts with functions and classes
- [ ] Work with NumPy arrays and Pandas DataFrames
- [ ] Load, clean, and transform datasets
- [ ] Split data into train/test sets
- [ ] Apply feature scaling and encoding
- [ ] Train a classification model (Logistic Regression, Random Forest)
- [ ] Train a regression model (Linear Regression)
- [ ] Evaluate models with appropriate metrics
- [ ] Save and load models with joblib/pickle

### Deep Learning
- [ ] Understand basic NN architecture
- [ ] Build a simple neural network with TensorFlow/Keras or PyTorch
- [ ] Implement a CNN for image classification
- [ ] Implement an LSTM/RNN for sequence data
- [ ] Know how to prevent overfitting (dropout, regularization)

### Linux & Shell
- [ ] Navigate file systems with command line
- [ ] Create, copy, move, and delete files/directories
- [ ] Use grep, find, and other search utilities
- [ ] Write basic shell scripts
- [ ] Understand file permissions
- [ ] Work with environment variables

### Git & GitHub
- [ ] Initialize a Git repository
- [ ] Stage and commit changes
- [ ] Create and merge branches
- [ ] Push/pull from remote repositories
- [ ] Resolve merge conflicts
- [ ] Create a .gitignore file

### AWS
- [ ] Create an AWS account
- [ ] Configure AWS CLI
- [ ] Upload/download files to/from S3
- [ ] Launch and connect to an EC2 instance
- [ ] Understand basic IAM concepts

### Docker
- [ ] Install Docker on your system
- [ ] Pull and run Docker images
- [ ] Create a Dockerfile
- [ ] Build Docker images
- [ ] Run containers with port mapping and volume mounts
- [ ] Use Docker Compose for multi-container applications

---

## 11. Quick Start Commands

### Setting Up Your Environment

```bash
# Create project directory
mkdir mlops-project
cd mlops-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install essential packages
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Initialize Git
git init
echo "venv/" >> .gitignore
echo "*.pkl" >> .gitignore
echo "*.csv" >> .gitignore
git add .
git commit -m "Initial commit"

# Create GitHub repo and push
git remote add origin https://github.com/username/mlops-project.git
git push -u origin main
```

### Test Your Setup

```python
# test_setup.py
import sys
print("Python version:", sys.version)

import numpy as np
print("NumPy version:", np.__version__)

import pandas as pd
print("Pandas version:", pd.__version__)

import sklearn
print("Scikit-learn version:", sklearn.__version__)

try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
except:
    print("TensorFlow not installed")

try:
    import torch
    print("PyTorch version:", torch.__version__)
except:
    print("PyTorch not installed")

print("\nAll essential libraries are installed!")
```

---

## Congratulations! 🎉

You now have a solid foundation to start the MLOps with Cloud Season 2 course. Remember:

1. **Practice regularly** - Code every day, even if just for 30 minutes
2. **Build projects** - Apply what you learn to real problems
3. **Join communities** - Engage with ML/MLOps communities on Discord, Reddit, Twitter
4. **Ask questions** - Don't hesitate to seek help when stuck
5. **Document your learning** - Keep notes and create a portfolio

**Good luck with your MLOps journey!** 🚀
