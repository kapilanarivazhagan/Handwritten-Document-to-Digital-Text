import streamlit as st
import tempfile
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import fitz  # PyMuPDF for PDF processing
import os

# Load the OCR model with caching and error handling
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = r"C:\Users\USER\Desktop\DEPLOY\ocr_model.h5"  # Update this path if needed
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("OCR Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# Function to preprocess the input image for OCR model
def preprocess_image(image):
    try:
        image = np.array(image.convert("L"))  # Convert to grayscale
        image = cv2.resize(image, (128, 32))  # Resize to model input size
        image = image / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        return None

# Function to decode predictions from the OCR model
def decode_predictions(predictions):
    char_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"  # Modify as needed
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    decoded_result = tf.keras.backend.ctc_decode(predictions, input_length=input_len)[0][0].numpy()
    
    decoded_text = ""
    for char_idx in decoded_result[0]:
        if char_idx != -1:
            decoded_text += char_list[char_idx]
    return decoded_text

# Function to extract images from PDF files
def extract_images_from_pdf(pdf_path):
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return images

# Function to handle file input selection
def display_input_form():
    input_type = st.sidebar.radio("Select input type:", ("Upload Image/PDF", "Enter File Path"))
    file_path = None

    if input_type == "Upload Image/PDF":
        st.sidebar.markdown("### Upload Image/PDF")
        uploaded_file = st.sidebar.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])
        
        if uploaded_file is not None:
            file_extension = ".jpg" if uploaded_file.type.startswith('image') else ".pdf"

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(uploaded_file.read())
                return temp_file.name

    elif input_type == "Enter File Path":
        file_path = st.sidebar.text_input("Enter the full file path:")
        if file_path and os.path.exists(file_path):
            return file_path
        elif file_path:
            st.error("File path does not exist. Please enter a valid path.")
            st.stop()

    return None

# Function to display extracted text and uploaded images
def display_results(images, extracted_text):
    st.subheader("Uploaded Image/Document:")
    for img in images:
        st.image(img, caption="Processed Image", use_column_width=True)

    st.subheader("Extracted Text:")
    st.info("\n".join(extracted_text) if extracted_text else "No text extracted.")
    st.success("OCR process completed successfully!")

# Main function to run the Streamlit app
def main():
    st.title("Handwritten Text Recognition with OCR")
    st.write("Upload an image or PDF to extract text.")

    file_path = display_input_form()

    if file_path:
        if file_path.endswith(".pdf"):
            images = extract_images_from_pdf(file_path)
        else:
            images = [Image.open(file_path)]

        extracted_texts = []
        for img in images:
            processed_img = preprocess_image(img)
            if processed_img is not None:
                prediction = model.predict(processed_img)
                recognized_text = decode_predictions(prediction)
                extracted_texts.append(recognized_text)

        display_results(images, extracted_texts)

if __name__ == "__main__":
    main()
