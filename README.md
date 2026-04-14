# Explainable AI (XAI) on MNIST Dataset
### Techniques: CNN + SHAP + LIME + Grad-CAM

This project implements Explainable AI (XAI) techniques on the MNIST handwritten digit dataset using a Convolutional Neural Network (CNN). The goal is to not just build an accurate model, but to **understand why** the model makes its predictions.

---

## 📁 Dataset

**MNIST in CSV format** — 70,000 handwritten digit images (28x28 pixels), split into:
- `mnist_train.csv` — 60,000 training samples
- `mnist_test.csv` — 10,000 test samples

> ⚠️ The dataset files are **not included** in this repo due to GitHub's file size limit.  
> 📥 Download them from Kaggle: [MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  
> After downloading, place both CSV files in the root of this project folder.

---

## 🧠 Model Architecture

A CNN built with TensorFlow/Keras:

| Layer | Output Shape | Params |
|-------|-------------|--------|
| Conv2D (32 filters, 3x3) | (26, 26, 32) | 320 |
| MaxPooling2D (2x2) | (13, 13, 32) | 0 |
| Conv2D (64 filters, 3x3) | (11, 11, 64) | 18,496 |
| MaxPooling2D (2x2) | (5, 5, 64) | 0 |
| Flatten | (1600) | 0 |
| Dense (128, ReLU) | (128) | 204,928 |
| Dropout (0.3) | (128) | 0 |
| Dense (10, Softmax) | (10) | 1,290 |

**Total Parameters:** 225,034  
**Test Accuracy:** ~99%

---

## 🔍 XAI Techniques Applied

### 1. SHAP (SHapley Additive exPlanations)
- **Global Explanation** — Pixel-level importance map showing which pixels matter most across the entire dataset
- **Local Explanation** — Per-sample force plots showing which pixels push a prediction up (red) or down (blue)

### 2. LIME (Local Interpretable Model-Agnostic Explanations)
- Highlights superpixel regions that positively or negatively contribute to a specific prediction
- Shows which parts of the digit the model focuses on

### 3. Grad-CAM (Gradient-weighted Class Activation Mapping)
- Visualizes CNN layer activations as heatmaps overlaid on the original image
- Shows which regions of the image the convolutional layers respond to most strongly

---

## 📊 Results & Visualizations

| Visualization | Description |
|--------------|-------------|
| `sample_images.png` | Sample digits from the dataset |
| `training_curves.png` | Accuracy and loss over epochs |
| `confusion_matrix.png` | Per-class prediction performance |
| `shap_global_importance.png` | Global pixel importance heatmap |
| `shap_image_plot.png` | SHAP local explanations per sample |
| `lime_explanation.png` | LIME superpixel explanation |
| `gradcam_visualization.png` | Grad-CAM heatmap overlays |
| `misclassified.png` | Examples the model got wrong |

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/RavirajSonar40/DLT_THEORY_ASSIGNMENT2.git
cd DLT_THEORY_ASSIGNMENT2
```

2. Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies
```bash
pip install tensorflow shap lime scikit-learn numpy pandas matplotlib seaborn scikit-image ipykernel scipy
```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and place `mnist_train.csv` and `mnist_test.csv` in the project folder

5. Open and run the notebook
```bash
jupyter notebook xai_mnist_assignment.ipynb
```

---

## 🛠️ Libraries Used

| Purpose | Library |
|---------|---------|
| Deep Learning | TensorFlow, Keras |
| XAI | SHAP, LIME |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Image Processing | scikit-image, scipy |
| ML Utilities | scikit-learn |

---

## 📌 Key Findings

- The CNN achieves **~99% test accuracy** on MNIST
- SHAP global map confirms the model correctly focuses on **center pixels** and ignores background corners
- Grad-CAM shows the CNN detects **edges and curves** of digits rather than full shapes
- Digits **4 and 9** are most commonly confused — XAI reveals they activate similar pixel regions
- LIME highlights that even small stroke differences (e.g., the top curve of a 9 vs 4) are key decision regions
