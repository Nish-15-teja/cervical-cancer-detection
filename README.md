# cervical-cancer-detection
Project Overview

This project presents an Explainable Artificial Intelligence (XAI) framework for automated cervical cancer cell classification using deep learning. The system classifies Pap smear cell images from the SIPaKMeD dataset into five different cervical cell categories. A CBAM (Convolutional Block Attention Module) integrated ResNet50 model is proposed and compared with baseline deep learning models to demonstrate improved classification performance and better feature learning.

The application also provides visual explanations using Grad-CAM, enabling users to understand which regions of the image influenced the model's prediction.

Dataset

Dataset: SIPaKMeD (Cervical Cell Dataset)

Total Images: 4049
Number of Classes: 5

Classes:

Immature Dysplastic (ImDys)
Koilocytotic (Koil)
Metaplastic (Meta)
Parabasal (Parab)
Superficial-Intermediate (Super)
Features
Deep Learning-based Cervical Cancer Cell Classification
ResNet50 Baseline Model
DenseNet121 Model
EfficientNet-B0 Model
Proposed CBAM-ResNet50 Attention Model
Transfer Learning
Data Augmentation
Grad-CAM Explainability
SHAP Feature Interpretation
Interactive Streamlit Web Application
Confidence Score Visualization
Model Comparison
Confusion Matrix
Performance Metrics
Project Workflow
Dataset Collection (SIPaKMeD)
Image Preprocessing
Data Augmentation
Train-Validation-Test Split
Model Development
ResNet50
DenseNet121
EfficientNet-B0
CBAM-ResNet50
Transfer Learning & Fine-Tuning
Model Evaluation
Explainable AI (Grad-CAM & SHAP)
Model Comparison
Final Prediction using Streamlit Application
Technologies Used
Python
PyTorch
Streamlit
OpenCV
NumPy
Matplotlib
Plotly
PIL
SHAP
Grad-CAM
Performance Evaluation

The models are evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
ROC Curve
Classification Report

The proposed CBAM-ResNet50 model achieves superior classification performance compared to the baseline ResNet50, DenseNet121, and EfficientNet-B0 models.

Explainable AI

To improve model interpretability, the project incorporates:

Grad-CAM for visualizing important image regions influencing predictions.
SHAP for understanding feature contributions and model decisions.

These techniques enhance transparency and trustworthiness in AI-assisted medical diagnosis.

Streamlit Application

The web application allows users to:

Upload a trained model (.pt/.pth)
Upload cervical cell images
Predict cell type
Display prediction confidence
Visualize Grad-CAM heatmaps
Compare model outputs

Repository Structure
cervical-cancer-detection/
│── app.py
│── Final_Minor_project.ipynb
│── requirements.txt
│── README.md
│── models/
│── outputs/
│── images/

