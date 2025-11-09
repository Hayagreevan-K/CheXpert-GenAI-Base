
# 🩻 CheXpert Base Model — GenAI + Grad-CAM + Continual Learning Ready

This project implements a **GenAI-powered medical imaging pipeline** using the **CheXpert dataset**, integrating:
- 🧠 **Deep Learning** for chest X-ray classification  
- 🔥 **Grad-CAM** visualization for interpretability  
- 💬 **OpenAI GPT** for automated radiology-style reporting  
- 🔄 **Continual Learning readiness** (end users can add new data)  
- 🌐 **Streamlit UI** for interactive visualization and reporting  

---

## ⚙️ Project Overview

The project follows a **multi-stage design** to ensure reproducibility and modularity:

### 🧩 Core Components
| Stage | Environment | Purpose |
|--------|--------------|----------|
| **1. Model Training (Base)** | 🧱 **Kaggle** | Model trained on CheXpert dataset (DenseNet121 fine-tuning) |
| **2. Sample Data Prep** | 🧱 **Kaggle** | Extracts & stores smaller sample images for testing |
| **3. OpenAI Integration** | ☁️ **Google Colab** | Generates radiology-style reports using GPT |
| **4. Visualization & UI** | 🌐 **Streamlit** | Frontend for uploading X-rays, visualizing Grad-CAM, and generating reports |
| **5. Continual Learning Support** | 🧩 **Optional (Colab)** | Allows users to add more data and retrain/fine-tune the model |

---

## 🧭 Pipeline Flowchart

A[Kaggle: CheXpert Dataset] --> B[Model Training (DenseNet121)]

B --> C[Save Outputs (.pth, .json, .csv)]

C --> D[Kaggle: Export Sample Images (subset of CheXpert)]

D --> E[Google Drive / Colab Integration]

E --> F[Colab: Load Base Model + OpenAI Key]

F --> G[Generate AI Radiology Reports]

G --> H[Streamlit: Visualization Interface]

H --> I[Grad-CAM Heatmap Overlay + OpenAI Text Report]

I --> J[End User Adds New Data → Fine-Tuning Ready]


# LINKS

STREAMLIT --- http://localhost:8501/
DRIVE --- https://drive.google.com/drive/folders/1-fy-eYzK0p0c2QAVRYzT6DgvLexPLAxN
DATASET USED --- https://www.kaggle.com/datasets/ashery/chexpert


