import streamlit as st

st.set_page_config(page_title="Mask Detection", layout="centered")

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

st.title("😷 CNN Model Inference for Mask Detection")

# تحميل النموذج
try:
    model = load_model('cnn_model .h5')  # تأكد إن الملف موجود في نفس المجلد
    st.success("✅ Model Loaded Successfully!")
    st.write("📋 Model Summary:")
    st.text(str(model.summary()))
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# أسماء الكلاسات
class_names = ['without_mask', 'with_mask']

# تحميل صورة من المستخدم
uploaded = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    invert_image = st.checkbox("🌓 Apply Invert (Use only for negative-style images)", value=False)

    img = Image.open(uploaded).convert("RGB")
    if invert_image:
        img = ImageOps.invert(img)

    img_resized = img.resize((128, 128))
    st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.write(f"📏 Image shape after processing: {img_array.shape}")

    if st.button("🔍 Predict"):
        try:
            prediction = model.predict(img_array)
            st.write(f"📊 Prediction output shape: {prediction.shape}")

            if prediction.shape == (1, 2):
                prediction = prediction[0]
                predicted_index = np.argmax(prediction)
                predicted_label = class_names[predicted_index]
                confidence = prediction[predicted_index]

                st.markdown("---")
                st.markdown(f"### ✅ Predicted Class: **{predicted_label}**")
                st.markdown(f"### 📈 Confidence: **{confidence:.2f}**")

                with st.expander("📊 Show Prediction Probabilities"):
                    for class_name, prob in zip(class_names, prediction):
                        st.markdown(f"- **{class_name}**: `{prob:.2f}`")

            elif prediction.shape == (1, 1):
                prob = float(prediction[0])
                predicted_label = class_names[0] if prob >= 0.5 else class_names[1]
                confidence = prob if prob >= 0.5 else 1 - prob

                st.markdown("---")
                st.markdown(f"### ✅ Predicted Class: **{predicted_label}**")
                st.markdown(f"### 📈 Confidence: **{confidence:.2f}**")

                with st.expander("📊 Show Prediction Probabilities"):
                    st.markdown(f"- **with_mask**: `{prob:.2f}`")
                    st.markdown(f"- **without_mask**: `{1 - prob:.2f}`")

            else:
                st.error(f"❌ Unexpected prediction shape: {prediction.shape}. Expected (1, 2) or (1, 1).")

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
