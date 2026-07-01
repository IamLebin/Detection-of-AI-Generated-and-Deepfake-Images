# Detection of AI-Generated and Deepfake Images (Metric Model)

A Streamlit web application that uses a deep learning model to detect whether an image is Real or Fake/AI-generated.

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Download the Model File
Due to file size limits on GitHub, the `model.h5` file is not included in this repository. 
- Download `model.h5` from your storage link (e.g., Google Drive / OneDrive).
- Place the `model.h5` file directly in the root of this project directory.

### 3. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Start the Streamlit server:
```bash
streamlit run app.py
```

---

## Slides & Presentation
The presentation slides for this project are located in the `reveal.js/` folder.
To view them, open `reveal.js/index.html` in your browser.
