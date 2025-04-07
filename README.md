# 🧠 GenOrNot – Real vs AI-Generated Image Detector

**GenOrNot** is a hybrid machine learning project designed to classify images as either **real** (captured by a camera) or **AI-generated** (created using tools like DALL·E, MidJourney, or Craiyon). It implements **two pipelines**: one using **handcrafted image features + PCA + Logistic Regression / Random Forest / KMeans / GMM / SVM**, and another using **deep features extracted via VGG16 + SVM**. The final model is deployed on a **Flask web app hosted on AWS**.

---

## 📌 Table of Contents

- [Motivation](#-motivation)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Handcrafted Feature Pipeline](#-handcrafted-feature-pipeline)
- [VGG + SVM Pipeline](#-vgg--svm-pipeline)
- [Deployment](#-deployment)


---

## 💡 Motivation

Generative AI is advancing rapidly, making it harder to distinguish real photos from synthetically created ones. This poses risks in areas like journalism, forensics, and media integrity. Our goal is to build a lightweight and efficient detection system that can identify these images with high accuracy using both classical and deep learning methods.

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- scikit-learn
- OpenCV, NumPy
- Flask (Web App)
- AWS EC2 (Deployment)

---

## 📂 Dataset

- **Real Images**: Captured using digital cameras
- **Fake Images**: Generated using DALL·E, MidJourney, and Craiyon
- Categories include: jungle, sea, mountain, etc.
- Preprocessing included resizing, label normalization (via fuzzy matching), and standardization

---

## 🔍 Methodology

The project uses **two complementary pipelines**:

1. **Pipeline 1** – Handcrafted features using (color, edge, texture)
2. **Pipeline 2** – Deep features using VGG16

Both approaches are evaluated to compare performance.

---

## 🧪 Handcrafted Feature Pipeline

1. **Feature Engineering**:
   - Extracted color histograms, Haralick texture features, edge density, and contour-related properties
   - LBP (Local Binary Pattern):
      Captures the texture of the grayscale version of the image by comparing the intensity of each pixel with its surrounding neighbors. LBP is rotation-invariant and efficient for texture classification.

   - FFT (Fast Fourier Transform):
      Converts the image into the frequency domain and extracts global frequency patterns. Real images and AI-generated ones often differ in the distribution of frequency components.

2. **Dimensionality Reduction**:
   - Applied PCA to reduce noise and compress features
3. **Classifier**:
   - Logistic Regression
4. **Hyperparameter Tuning**:
   - Regularization parameter `C` adjusted for optimal bias-variance tradeoff

---

## 🧠 VGG + SVM Pipeline

1. **Feature Extraction**:
   - Used pre-trained **VGG16** model (without top layers) to extract feature maps
   - Flattened the convolutional outputs for classification
2. **Classifier**:
   - Support Vector Machine with RBF kernel
3. **Hyperparameter Tuning**:
   - `C` and `gamma` parameters were tuned using grid search
4. **Performance**:
   - Outperformed handcrafted method in both accuracy and generalization

---

## 🚀 Deployment

- A Flask-based web app is developed to let users upload images and receive predictions
- Hosted on **AWS EC2 instance** for public access
- Includes a simple UI and backend model inference

---

### Clone the Repo
```bash
git clone https://github.com/Mmb0129/Real-Vs-Fake-Detector.git

