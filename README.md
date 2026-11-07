# 🫁 Pneumonia Detection and Localization in Chest X-Rays

**A hybrid AI framework (classical ML + deep learning) for detecting and localizing pneumonia in chest X-ray images, with explainable predictions and an interactive Streamlit demo.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-RSNA%20Pneumonia-0A66C2)
![AUC](https://img.shields.io/badge/ResNet--50%20AUC--ROC-0.8745-success)
![License](https://img.shields.io/badge/License-MIT-green)

> ⚠️ **Disclaimer:** This project is for research and educational purposes only. It is **not** a medical device and must not be used for clinical diagnosis or treatment decisions.

---

## 📌 Overview

Pneumonia causes millions of deaths worldwide, and radiologists face high workloads, human error, and inter-observer variability. This project builds an AI-assisted diagnosis system that:

- **Detects** pneumonia from chest X-rays (DICOM)
- **Localizes** infected regions with bounding boxes
- **Explains** its predictions with Grad-CAM heatmaps
- **Compares** classical machine learning against deep learning

The system combines the **interpretability** of classical ML with the **performance** of deep learning, targeting both clinical trust and diagnostic accuracy.

## ✨ Key Features

- 🩻 **Native DICOM support** with a full medical-imaging preprocessing pipeline
- 🧠 **Dual-branch deep learning design:** ResNet-50 answers *"Is pneumonia present?"*, Faster R-CNN answers *"Where is it?"*
- 🔍 **Explainability** through Grad-CAM heatmaps and bounding-box evidence
- ⚖️ **Imbalance-aware training** with Focal Loss and threshold optimization
- 🧪 **Classical ML baseline** with lung segmentation, handcrafted features, and 9 compared models
- 🖥️ **Streamlit web app** for instant upload-and-classify inference on CPU

## 🏗️ System Architecture

```
                          Chest X-ray (DICOM)
                                  │
                         DICOM Preprocessor
          (HU conversion → lung windowing → CLAHE → normalization)
                                  │
              ┌───────────────────┴────────────────────┐
              ▼                                        ▼
        Classical ML branch                     Deep Learning branch
              │                                        │
   Lung segmentation (Otsu)                 ┌───────────┴────────────┐
   32 handcrafted features                  ▼                        ▼
   SelectKBest + StandardScaler      ResNet-50 classifier   Faster R-CNN + FPN
   SMOTE balancing                   "Is pneumonia present?" "Where is it?"
   9 ML models + tuning              + Grad-CAM              + bounding boxes
```

## 📊 Dataset

**RSNA Pneumonia Detection Challenge**, 26,684 chest X-ray images in DICOM (`.dcm`) format.

Each sample contains the X-ray image, patient metadata, a label, and bounding-box annotations (pneumonia cases only).

| Class | Share |
|---|---|
| Normal | 33.2% |
| Lung Opacity (pneumonia) | 22.5% |
| No Lung Opacity / Not Normal | 44.3% |

For binary classification, pneumonia is the positive class: **6,012 pneumonia (22.5%)** vs **20,672 non-pneumonia (77.5%)**, an imbalance of roughly **3.4 : 1**.

**Localization statistics:** 9,555 bounding boxes across 6,012 pneumonia patients (1.59 per patient on average). About 55% of cases have 2 boxes (often bilateral infection), about 43% have 1 box, and a few have 3 or more. The mean box area is about 77,523 px², so opacities are generally large and diffuse.

**Dataset challenges:** class imbalance, low contrast, overlapping anatomical structures, multiple lesions per image, variable annotation quality, and generalization to unseen data.

**Key EDA insight:** mean pixel intensity is almost identical between classes (Normal 0.921 vs Pneumonia 0.917), and the intensity distributions overlap strongly. Simple statistical features are insufficient, which motivates texture features and deep learning. The largest class is non-pneumonia abnormalities, so the model must learn pneumonia-specific features, not just "normal vs abnormal."

## 🔧 Preprocessing and Augmentation

**DICOM preprocessing**

1. **DICOM reading:** extract raw pixel values
2. **HU conversion:** apply RescaleSlope and Intercept
3. **Lung windowing:** keep lung-relevant intensities, remove irrelevant tissue
4. **CLAHE:** local contrast enhancement that highlights opacities without amplifying noise
5. **Normalization:** scale values to [0, 1]

**Augmentation pipeline** (preserves the medical relevance of pneumonia patterns)

| Technique | Parameters | Probability |
|---|---|---|
| Horizontal flip | none | 0.5 |
| Rotation | ±10° | 0.3 |
| Translation | ±10% shift | 0.3 |
| Brightness / contrast | ±20% | 0.4 |
| Gaussian noise | σ = 0.01 | 0.2 |

## 🧪 Classical ML Pipeline

| Step | Details |
|---|---|
| **Lung segmentation** | Otsu thresholding → morphological opening (5×5 elliptical) → closing (20×20 elliptical) → keep the 2 largest connected components (left and right lung) |
| **Feature extraction** | 32 handcrafted features per patient: intensity statistics (mean, std, skewness, kurtosis, min, max, median, IQR), shape features (area, perimeter, circularity, left/right ratio, eccentricity), and GLCM texture features (6 properties × 4 angles, 64 gray levels) |
| **Feature selection and scaling** | `SelectKBest` + `StandardScaler` |
| **Dimensionality reduction** | PCA was tested but gave no improvement and slightly reduced accuracy, so it was dropped |
| **Class balancing** | SMOTE oversampling of pneumonia cases |
| **Models** | Logistic Regression, SVM, Random Forest, LightGBM, Gradient Boosting, AdaBoost, XGBoost, CatBoost, Extra Trees, each with hyperparameter optimization |

Every experiment ran on both segmented (mask on) and unsegmented (mask off) images. The final feature table has 18,678 training rows × 32 features + label.

## 🧠 Deep Learning Pipeline

### Branch 1: ResNet-50 classifier

- **Backbone:** ImageNet-pretrained ResNet-50 with frozen early layers (fewer trainable parameters, less overfitting, low-level features preserved)
- **Head:** custom classification head (`Linear(2048→512) → ReLU → Dropout(0.5) → Linear(512→2)`)
- **Loss:** Focal Loss (`α = 0.25`, `γ = 2`) to down-weight easy negatives and improve minority-class recall
- **Optimization:** AdamW + OneCycleLR, 12 epochs
- **Threshold tuning:** the default 0.5 was suboptimal, and the best decision threshold was **0.320**. In medicine, missing pneumonia is more dangerous than a false alarm.
- **Training behavior:** validation AUC-ROC rose steadily from 0.863 to 0.877 with no overfitting signal; early stopping triggered at epoch 12.

### Branch 2: Faster R-CNN detector

- **Backbone:** ResNet-50 + Feature Pyramid Network (FPN), so both small opacities and large diffuse infiltrates are detected
- **Multi-task learning:** classification, bounding-box regression, and region proposal generation (RPN)
- **Loss:** `L = L_cls + λ_box · L_box + L_RPN`
- **Optimization:** SGD with momentum, differential learning rates (pretrained backbone learns slowly, detection head learns faster), 8 epochs, final validation loss 0.1289

## 📈 Results

### ResNet-50 (classification)

| Metric | Score |
|---|---|
| Accuracy | 0.8059 |
| AUC-ROC | 0.8745 |
| F1-score | 0.6394 |
| Precision (PPV) | 0.5499 |
| Recall (sensitivity) | 0.7639 |
| Specificity | 0.8181 |
| NPV | 0.9225 |
| Average precision | 0.6820 |
| Optimal threshold | 0.320 |
| Inference time | 16.6 ms / image |

High recall means most pneumonia cases are caught, and high NPV means normal patients can be excluded reliably. This makes the model well suited to **triage and screening**.

### Faster R-CNN (detection and localization)

| Metric | Value |
|---|---|
| mAP@0.5 | 0.1011 |
| mAP@0.3 | 0.2606 |
| Mean IoU | 0.4452 |
| Inference time | 90.6 ms / image (~11 FPS) |

mAP@0.5 is modest because of the short 8-epoch training, noisy radiologist annotations, and severe class imbalance. Performance improves at relaxed IoU thresholds, and a mean IoU of about 0.45 shows substantial overlap with ground truth.

### Model comparison

| Criterion | ResNet-50 | Faster R-CNN |
|---|---|---|
| Task | Classification | Detection + localization |
| Speed | 16.6 ms | 90.6 ms |
| Localization | No | Yes |
| Explainability | Grad-CAM | Bounding boxes |
| Best metric | AUC = 0.8745 | IoU = 0.4452 |

The two branches complement each other: ResNet-50 gives fast screening with strong diagnostic confidence, and Faster R-CNN gives spatial localization and visual evidence for clinicians.

## 🔍 Explainability (Grad-CAM)

Black-box predictions are hard for clinicians to trust, so Grad-CAM heatmaps show where the CNN is looking.

- **Positive cases:** attention concentrates on opacity regions, mainly the lower and middle lung zones
- **Negative cases:** diffuse attention with no pathological hotspots

## 🖥️ Streamlit Demo

The included `app.py` serves the **ResNet-50 classification branch**: upload a DICOM, PNG, or JPEG chest X-ray and get a predicted class with a confidence score.

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**`requirements.txt`**

```
streamlit
torch
torchvision
numpy
Pillow
pydicom
```

### Add the model weights

Place the trained checkpoint (`resnet50_best.pt`) in the project root:

```
.
├── app.py
├── resnet50_best.pt     # trained ResNet-50 weights
├── requirements.txt
└── README.md
```

### Run

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

### Configuration

| Variable | Description | Value |
|---|---|---|
| `NUM_CLASSES` | Number of output classes; must match training | `2` |
| `CLASS_NAMES` | Labels in training index order | `["Normal", "Pneumonia"]` |

## 🗂️ Project Structure

```
.
├── app.py               # Streamlit demo (ResNet-50 inference)
├── resnet50_best.pt     # Trained checkpoint
├── requirements.txt
├── notebooks          # Training and EDA notebooks (optional)
└── README.md
```

## ⚠️ Limitations

- Performance drops on low-quality or portable chest X-rays
- Deep learning models need GPU resources for real-time inference at scale
- Class imbalance (~22.5% positive cases) may still bias predictions
- The fusion strategy between the ML and DL branches is not fully optimized
- RSNA annotations vary between radiologists, and bounding boxes are coarse
- No external validation on independent datasets has been performed

## 🛣️ Future Work

- Train Faster R-CNN for more epochs to improve localization accuracy
- Tune learning rate, batch size, anchor box scales, and confidence thresholds
- Use advanced learning-rate schedulers for more stable convergence
- Train on multi-institutional datasets to reduce domain shift
- Improve performance on low-quality X-rays
- Optimize the ML + DL fusion strategy
- Add Faster R-CNN detection and Grad-CAM visualization to the Streamlit app

## 🔒 Security and Privacy Notes

- `torch.load(..., weights_only=False)` can execute arbitrary code from untrusted pickle files. **Only load checkpoints you trained or fully trust.**
- DICOM files can contain patient-identifiable metadata. Avoid uploading real patient data to any publicly hosted deployment.


**Thynk Unlimited**


## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

## 🙏 Acknowledgements

- [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) for the dataset
- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/) for ResNet-50 and Faster R-CNN
- [Streamlit](https://streamlit.io/) for the web framework
- [pydicom](https://pydicom.github.io/) for DICOM parsing
