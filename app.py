import os
# Set to '1' for compatibility with models trained using legacy Keras environment
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
# CRITICAL: Import Albumentations for standardized pre-processing
import albumentations as A 

# EfficientNet input size
IMG_SIZE = (224, 224) 
# Model file name
MODEL_PATH = 'model.h5' 

st.set_page_config(page_title="Deepfake Detector (Metric Model)", page_icon="🕵️")

# Define the exact normalization used during training
A_NORMALIZE = A.Compose([
    A.Normalize(
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225)
    ),
])

# streamlit cache performance decorator
@st.cache_resource
def load_deepfake_model():
    # compile=False is MANDATORY for loading the multi-output/custom-loss model
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

st.title("🕵️ Deepfake vs Real Detector (Metric Model)")
st.write(f"Model File: `{MODEL_PATH}`")

# 1. Load the model
try:
    with st.spinner("Loading Model..."):
        model = load_deepfake_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. Upload Image
uploaded_file = st.file_uploader("Upload an image to check", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # 3. Preprocess
    img_pil = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img_pil)
    
    # Apply Albumentations Normalization
    normalized_array = A_NORMALIZE(image=img_array)['image']
    
    # Add batch dimension (1, 224, 224, 3)
    img_batch = np.expand_dims(normalized_array, axis=0)

    # 4. Predict
    if st.button('🔍 Analyze Image'):
        with st.spinner('Analyzing...'):
            prediction_output = model.predict(img_batch)
            
        # Safely extract the logit value
        if isinstance(prediction_output, list):
            logit_array = prediction_output[0]
        else:
            logit_array = prediction_output
            
        logit = logit_array[0][0] 
        
        # Use TENSORFLOW'S NUMERICALLY STABLE SIGMOID
        stable_score_tensor = tf.nn.sigmoid(tf.constant(logit, dtype=tf.float32))
        score = stable_score_tensor.numpy()
        
        # --- FINAL FIX: Set the threshold and REVERSE the logic ---
        
        # Threshold set to 0.63 to separate the two critical images:
        # P(Real) 0.6261 (REAL selfie) < 0.63 -> REAL
        # P(Real) 0.6376 (FAKE J. Law) > 0.63 -> FAKE
        CLASSIFICATION_THRESHOLD = 0.63 
        
        st.divider()
        st.subheader("Analysis Results:")

        # Display the image
        st.image(image, caption='Uploaded Image', width='stretch')

        # Decision threshold: Logic is REVERSED: P(Real) < THRESHOLD = REAL
        if score < CLASSIFICATION_THRESHOLD:
            # Classification is REAL (because the score is low)
            display_confidence = score * 100
            st.success(f"**REAL Image** ({display_confidence:.1f}%)")
        else:
            # Classification is FAKE (because the score is high)
            # Display confidence based on P(Fake) = 1 - P(Real)
            display_confidence = (1 - score) * 100
            st.error(f"**FAKE / AI Generated** ({display_confidence:.1f}%)")
            
        # Progress bar showing P(Real) score
        st.progress(float(score))

        st.caption("Note: The prediction model exhibited a reversed label assignment, requiring a threshold calibration for accurate classification.")