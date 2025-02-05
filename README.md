# Handwritten-Document-to-Digital-Text
Develop a classifier that recognizes handwritten text and converts it into digital text using OCR (Optical Character Recognition) technology. The system will take a handwritten PDF file as input and output the converted digital text in a .txt file format.

## Project Overview
This project aims to build an OCR (Optical Character Recognition) system capable of recognizing handwritten text in images and PDFs. The model uses a combination of **Convolutional Neural Networks (CNNs)**, **Long Short-Term Memory (LSTM)** networks, and **CTC (Connectionist Temporal Classification)** loss to achieve state-of-the-art performance in handwritten text recognition. The system is deployed using **Streamlit** for easy user interaction.

## Features
- Upload handwritten images or PDF files for OCR.
- Display the extracted text from images or PDFs.
- Real-time model predictions on uploaded files.
- High accuracy in recognizing handwritten text.

## Requirements
Before running the application, ensure that you have the following dependencies installed:

- Python 3.8 or later
- TensorFlow
- Streamlit
- OpenCV
- Pillow
- numpy
- PyMuPDF (for handling PDFs)
  
Install the required dependencies by running:

```bash
pip install -r requirements.txt

## Acknowledgements

- The architecture is inspired by various OCR research papers and implementations using CNN, LSTM, and CTC.
- Thank you to the community for providing datasets like IAM Handwriting for training the model.
