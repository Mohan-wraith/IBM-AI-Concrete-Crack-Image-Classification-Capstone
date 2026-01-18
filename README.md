# 🏗️ Concrete Crack Detection System
### IBM AI Engineering Capstone Project

**Author:** M.MOHAN | **Date:** November 2025

---

## 📖 Executive Summary
This project delivers a production-ready Deep Learning solution for automated structural health monitoring. By leveraging Transfer Learning with **VGG16** and **ResNet50**, the system detects surface cracks in concrete structures with **>99.5% accuracy**.

The final deployed model uses a custom-optimized **VGG16 architecture** that achieves state-of-the-art performance while remaining **40% lighter** than standard implementations, making it suitable for edge deployment on mobile inspection devices.

## 📊 Key Results
| Model Architecture | Accuracy | Precision | Recall | File Size | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VGG16 (Optimized)** | **>99.5%** | **1.00** | **0.99** | **~57 MB** | 🏆 **Production** |
| ResNet50 | 100% | 1.00 | 1.00 | ~96 MB | Archived |

> **Optimization Note:** The VGG16 model was modified to use **Global Average Pooling (GAP)** instead of standard flattening, significantly reducing the parameter count without sacrificing spatial feature extraction capabilities.

## 🧠 Explainable AI (Grad-CAM)
To ensure reliability, **Grad-CAM (Gradient-weighted Class Activation Mapping)** was implemented to visualize the model's decision-making process.
* **Positive Detection:** The model accurately highlights the specific crack geometry (Red/Yellow regions).
* **Negative Detection:** The model focuses on broad surface textures, confirming no specific "crack features" were activated.

## 📥 Model Download
Due to GitHub's file size limits, the trained model weights are hosted externally.
* **VGG16 Final Model (.keras):** [🔗 Link to your model or 'Available upon request']

## 🛠️ Tech Stack
* **Core Framework:** TensorFlow / Keras
* **Data Processing:** NumPy, Pandas, OpenCV
* **Visualization:** Matplotlib, Seaborn
* **Techniques:** Transfer Learning, Data Augmentation, Early Stopping

## 🚀 Usage Guide
# 🏗️ Concrete Crack Detection System
### IBM AI Engineering Capstone Project

**Author:** M.MOHAN | **Date:** November 2025

---

## 📖 Executive Summary
This project delivers a production-ready Deep Learning solution for automated structural health monitoring. By leveraging Transfer Learning with **VGG16** and **ResNet50**, the system detects surface cracks in concrete structures with **>99.5% accuracy**.

The final deployed model uses a custom-optimized **VGG16 architecture** that achieves state-of-the-art performance while remaining **40% lighter** than standard implementations, making it suitable for edge deployment on mobile inspection devices.

## 📊 Key Results
| Model Architecture | Accuracy | Precision | Recall | File Size | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VGG16 (Optimized)** | **>99.5%** | **1.00** | **0.99** | **~57 MB** | 🏆 **Production** |
| ResNet50 | 100% | 1.00 | 1.00 | ~96 MB | Archived |

> **Optimization Note:** The VGG16 model was modified to use **Global Average Pooling (GAP)** instead of standard flattening, significantly reducing the parameter count without sacrificing spatial feature extraction capabilities.

## 🧠 Explainable AI (Grad-CAM)
To ensure reliability, **Grad-CAM (Gradient-weighted Class Activation Mapping)** was implemented to visualize the model's decision-making process.
* **Positive Detection:** The model accurately highlights the specific crack geometry (Red/Yellow regions).
* **Negative Detection:** The model focuses on broad surface textures, confirming no specific "crack features" were activated.

## 📥 Model Download
Due to GitHub's file size limits, the trained model weights are hosted externally.
* **VGG16 Final Model (.keras):** [🔗 Link to your model or 'Available upon request']

## 🛠️ Tech Stack
* **Core Framework:** TensorFlow / Keras
* **Data Processing:** NumPy, Pandas, OpenCV
* **Visualization:** Matplotlib, Seaborn
* **Techniques:** Transfer Learning, Data Augmentation, Early Stopping

## 🚀 Usage Guide

### 1. Installation
```bash
pip install tensorflow opencv-python matplotlib numpy
```

###2. Inference Script
```Python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Load the optimized model
model = load_model('vgg16_concrete_final.keras')

def predict_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    # Preprocess (VGG16 specific)
    img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img_batch.copy())

    prediction = model.predict(img_preprocessed)
    confidence = prediction[0][1]

    if confidence > 0.5:
        return f"⚠️ CRACK DETECTED ({confidence:.2%} confidence)"
    else:
        return f"✅ Surface Intact ({1-confidence:.2%} confidence)"

# Test
print(predict_image('test_wall.jpg'))
```

## 📂 Repository Structure
```text
├── IBM AI Concrete Crack Image Classification.ipynb  # Main Source Code
├── test_predictions.csv                              # Raw Inference Results
├── README.md                                         # Project Documentation
└── (External Link) vgg16_concrete_final.keras        # Trained Model Weights
```

### 📜 License
This project is open-source and available under the MIT License.
