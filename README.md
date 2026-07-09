# 🎓 Emotion Detection & Learning Support Engine

## 📌 Project Overview

The **Emotion Detection & Learning Support Engine** is an AI-powered educational support platform that analyzes a student's learning challenges and predicts their emotional state using Natural Language Processing (NLP) and Deep Learning. Based on the detected emotion, the system provides personalized learning guidance to improve the student's learning experience.

The project leverages a **BiLSTM (Bidirectional Long Short-Term Memory)** model for emotion classification and integrates an intuitive **Streamlit** interface to deliver real-time predictions and AI-assisted learning recommendations.

---

## 🚀 Features

* 😊 Emotion Detection from student text input
* 📊 Emotion Probability Visualization
* 📈 Prediction Confidence Score
* 💡 Personalized Learning Advice
* 📚 Study Field Selection
* 🤖 AI Learning Assistant
* 📂 CSV Logging for Analytics
* 🎨 Interactive and Responsive Streamlit Dashboard
* 🔍 Text Preprocessing and Cleaning
* ⚡ Real-time Emotion Prediction

---

## 🎯 Problem Statement

Students often experience emotions such as confusion, frustration, boredom, curiosity, or confidence while learning. These emotions significantly impact learning outcomes but are rarely identified automatically.

This project aims to detect students' emotional states from textual descriptions of their learning challenges and provide personalized learning support, enabling a more engaging and effective educational experience.

---

## 💻 Technologies Used

* Python 3.11
* Streamlit
* TensorFlow / Keras
* BiLSTM Deep Learning Model
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Plotly
* Pickle
* Regular Expressions (Regex)

---

## 🧠 AI Model

The system uses a trained **Adaptive BiLSTM Model** for emotion classification.

### Supported Emotions

* 😎 Confident
* 😕 Confused
* 🤔 Curious
* 😣 Frustrated

The model predicts:

* Primary Emotion
* Confidence Score
* Emotion Probability Distribution
* Personalized Learning Advice

---

## 📂 Project Structure

```text
EmotionDetector/
│
├── pages/app.py
├── requirements.txt
├── emotion_response_mapping.csv
├── test_setup.txt
├── data/
│   └── processed/
│   └── raw/
│
├── models/
│   └── bilstm_student_adaptive.keras
│
├── src/
│   ├── model.py
│   └── predict.py
│
└── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/EmotionDetector.git
```

### Move into the Project Folder

```bash
cd EmotionDetector
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## 📋 How to Use

1. Launch the application.
2. Select your study field.
3. Enter your learning challenge.
4. Click **Detect Emotion**.
5. View:

   * Detected Emotion
   * Prediction Confidence
   * Emotion Probability Chart
   * Personalized Learning Advice

---

## 📊 Sample Output

The application displays:

* Detected Emotion
* Confidence Percentage
* Emotion Distribution Graph
* AI-generated Learning Guidance

---
## Mixed Emotion
input:
I understand the basics, but I'm still confused about recursion and also curious to learn more.
---

## 📈 Future Enhancements

* 🌍 Multilingual Emotion Detection
* 🎤 Voice-based Emotion Analysis
* 😊 Facial Expression Recognition
* ☁️ Cloud Deployment
* 📱 Mobile Application
* 👨‍🏫 Instructor Dashboard
* 📊 Advanced Analytics Dashboard
* 🤖 Gemini AI Integration for Dynamic Responses

---

## 🎯 Applications

* Smart Learning Platforms
* Online Education Systems
* E-learning Portals
* Student Mental Well-being Support
* Personalized Tutoring Systems
* Academic Performance Monitoring

---

## 👥 Project Team

Developed as part of the **Google Cloud Generative AI Internship** under **SmartBridge**.

---

## 📄 License

This project is developed for educational and academic purposes as part of the SmartBridge Internship Program.

---

## ⭐ Acknowledgements

* SmartBridge
* Google Cloud
* TensorFlow
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Open Source Community
