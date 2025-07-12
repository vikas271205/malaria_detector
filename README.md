# 🦠 Malaria Detector

A web-based deep learning application to detect malaria from cell images.

## 🧠 Model
- Trained on parasitized and uninfected cell images
- Accuracy: ~93%
- TensorFlow GPU acceleration enabled

## 🖥 Tech Stack
- **Backend**: Flask + TensorFlow
- **Frontend**: React
- **Model Input**: 64x64 RGB images

## 🧪 How to Run

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
