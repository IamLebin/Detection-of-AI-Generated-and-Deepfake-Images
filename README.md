# Detection-of-AI-Generated-and-Deepfake-Images (Metric Model)

A Streamlit web application that uses a deep learning model to detect whether an image is Real or Fake (AI-generated). This project was developed as part of a Machine Learning course assignment.

---

## 👥 Group Members
* **Chin Lok Bin** 
* **Lew Jia Jia** 
* **Tham Min Cong** 

---

## 📄 Project Report & Slides
* **Project Report:** [UECS2523_ML_Assignment Report_Group5.pdf](UECS2523_ML_Assignment%20Report_Group5.pdf) (Includes detailed methodology, data analysis, training pipeline, and results discussion).
* **Model Training Script:** [train_deepfake.py](train_deepfake.py) (The complete training pipeline script including face detection, augmentations, and model fine-tuning).
* **Presentation Slides:** Located in the [reveal.js/](reveal.js/) folder. To view them, open `reveal.js/index.html` in your browser.


---

## ⚙️ Model Architecture & Methodology

```text
                  [Input Image]
                        │
                        ▼
             [MTCNN Face Detection]   <-- Detects & crops face to 224x224
                        │
                        ▼
               [Albumentations]       <-- Applies normalization & augmentation
                        │
                        ▼
              [EfficientNetV2B3]      <-- Deep feature extraction (Transfer Learning)
                        │
                        ▼
              [Sigmoid Classifier]    <-- Classification logit output
                        │
                        ▼
             [Threshold Calibration]  <-- Custom 0.63 threshold correction
                        │
                        ▼
                 [Real vs Fake]
```

### Key Components:
1. **Preprocessing (MTCNN):** Eliminates background noise by detecting, cropping, and centering the primary face in every image, resizing it to a standard `224x224` format.
2. **Augmentation (Albumentations):** Normalizes inputs and applies dynamic transformations (horizontal flips, brightness/contrast shifts, blur, CoarseDropout) to make the model resilient against real-world variations.
3. **Core Model (EfficientNetV2B3):** Utilizes a pre-trained EfficientNetV2B3 backbone.
   * **Phase 1 (Head Training):** The base model is frozen; only the top classification layers are optimized.
   * **Phase 2 (Fine-Tuning):** The entire convolutional base is unfrozen and trained at an ultra-low learning rate (`5e-7`) to specialize in fine-grained deepfake artifacts.

### Datasets Used:
We combined two major Kaggle datasets to maximize generalization:
* [Deepfake Image Detection Dataset](https://www.kaggle.com/datasets/saurabhbagchi/deepfake-image-detection)
* [DeepDetect-2025](https://www.kaggle.com/datasets/ayushmandatta1/deepdetect-2025?select=ddata)

---

## 🚀 Setup & Execution Instructions

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd "your_path" 
```

### 2. Download the Model File
Due to file size limits on GitHub, the weight file `model.h5` is excluded from the repository.
* Download `model.h5` from your storage link (e.g., Google Drive / OneDrive).
* Place the `model.h5` file directly in the root of the cloned directory.

### 3. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Run the local server:
```bash
streamlit run app.py
```
This will automatically launch the web interface in your default browser.
