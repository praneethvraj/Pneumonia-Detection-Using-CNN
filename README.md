
# Pneumonia Detection from Chest X-Rays using Convolutional Neural Networks (CNN)

This project builds and evaluates a deep learning model to detect **Pneumonia** from chest X-ray images using **Convolutional Neural Networks (CNNs)**. The goal is to accurately classify images as either **NORMAL** or **PNEUMONIA**.

---

## 🩺 Project Objective

To develop a binary image classification model that distinguishes between:
- **NORMAL** lungs
- **Lungs infected with PNEUMONIA**

We leverage **TensorFlow/Keras** and techniques like **data augmentation**, **class weighting**, and **hyperparameter tuning** using **Optuna** and **MLflow** for experiment tracking.

---

## 📁 Dataset

We use the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) provided by Guangzhou Women and Children's Medical Center.

```
dataset/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## 🧠 Model Architecture

- Input layer (Rescaled to 224x224 or 128x128)
- Convolutional layers with ReLU
- MaxPooling layers
- Dropout for regularization
- Batch Normalization
- Fully Connected Dense layers
- Output layer (Sigmoid for binary classification)

---

## ⚙️ Features

- 🏷️ **Binary Classification** (NORMAL vs PNEUMONIA)
- ⚖️ **Class Balancing** using:
  - Class Weights
  - Data Augmentation (only on NORMAL class to handle imbalance)
- 📈 **Metrics**: Accuracy, Precision, Recall, F1-score
- 🔍 **Confusion Matrix** and classification report
- 🧪 **Hyperparameter Tuning** with [Optuna](https://optuna.org/)
- 📊 **Experiment Tracking** with [MLflow](https://mlflow.org/)

---

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/praneethvraj/pneumonia-detection-using-cnn.git
cd pneumonia-detection-using-cnn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Prepare dataset

Download the dataset and place it in the `dataset/` folder with proper train/val/test split.


 use the Jupyter notebook:

```bash
jupyter notebook pneumonia_detection-using-vgg-16-cnn.ipynb
```



Generates confusion matrix and classification report.

---

## 📊 Sample Output

**Confusion Matrix**

|               | Predicted NORMAL | Predicted PNEUMONIA |
|---------------|------------------|----------------------|
| Actual NORMAL |       167        |          8           |
| Actual PNEUMONIA |     54        |         486          |

**Classification Report**
```
               precision    recall  f1-score   support

      NORMAL       0.76      0.95      0.84       175
   PNEUMONIA       0.98      0.90      0.94       540

    accuracy                           0.91       715
   macro avg       0.87      0.93      0.89       715
weighted avg       0.93      0.91      0.92       715
```

---



---

## 🔧 Future Improvements

- Integrate transfer learning (e.g., ResNet, EfficientNet)
- Deploy model as a web app using Streamlit or Flask
- Handle multi-class classification (e.g., COVID-19, Tuberculosis)

---

## 🧪 Experiments Tracked with MLflow

- Epochs, learning rate, optimizer
- Loss and accuracy
- Best hyperparameters via Optuna
- Confusion matrix artifacts

---

## 📜 License

MIT License. See `LICENSE` for details.

---



## 💡 Contact

Feel free to reach out via [Email](praneethvraj@gmail.com) or open an issue in this repo.
