# Fabric-detection-system
# 🧵 Advanced Industrial Fabric Defect Detection System

This project presents a real-time intelligent system for detecting defects in industrial fabric using advanced computer vision techniques. It integrates OpenCV, scikit-image, and optionally deep learning, wrapped in a user-friendly **Streamlit dashboard**.

## 🚀 Project Overview

Manual inspection in textile quality control is prone to errors and inefficiencies. This project automates the detection of:
- Holes and tears
- Stains and discolorations
- Weaving pattern issues
- Color inconsistencies
- Structural anomalies (folds, misalignments)

All detections are visualized interactively, offering real-time feedback and quality approval decisions.

---

## 🛠️ Features

- 📷 Upload-based fabric image inspection
- 📊 Texture analysis using **GLCM**, **LBP**, and **Gabor filters**
- 🧠 Structural inspection via **FFT**, **Sobel**, and **Hough Line Transform**
- 🎨 Color variation detection using **HSV stats** and **KMeans**
- 🧩 Contour and shape-based defect classification (circularity, solidity)
- 📈 Interactive quality metrics and visualization
- ✅ Defect summary with approval/rejection logic
- 🧪 Optional: CNN integration for learned defect classification (stub provided)

---

## 🧪 Technologies Used

- Python 3.8+
- Streamlit
- OpenCV
- scikit-image
- NumPy, Matplotlib
- TensorFlow / Keras (for future CNN integration)

---



