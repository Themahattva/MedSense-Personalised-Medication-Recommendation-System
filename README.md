# 💊 MedSense: Personalised Medication Recommendation System

> An ML-powered system that recommends medications based on a patient's symptoms and predicted disease — combining multiple classifiers for robust, data-driven healthcare assistance.

---

## 📌 Overview

In clinical practice, medication selection often depends on individual patient profiles — symptoms, severity, allergies, and prior history. This project builds a **personalized medication recommendation pipeline** that:

1. Takes patient symptoms as input
2. Predicts the likely disease using trained ML classifiers
3. Recommends appropriate medications, precautions, diet, and workout advice based on the prediction

The goal is to provide an **end-to-end decision-support tool** that could assist healthcare professionals or patients in understanding their condition and potential treatments — not a replacement for professional medical advice.

---

## 🧠 How It Works

```
Patient Symptoms (input)
        ↓
  Feature Encoding
        ↓
 ┌──────────────────────────────────────┐
 │   Ensemble of ML Classifiers         │
 │   - Support Vector Machine (SVM)     │
 │   - Random Forest                    │
 │   - Naive Bayes                      │
 │   - K-Nearest Neighbors (KNN)        │
 │   - Gradient Boosting / Decision Tree│
 └──────────────────────────────────────┘
        ↓
  Disease Prediction
        ↓
  Lookup: Medications + Precautions + Diet + Workout
        ↓
  Personalized Recommendation Output
```

---

## 📊 Performance

| Classifier | Accuracy |
|---|---|
| Support Vector Machine | ~100% (on training split) |
| Random Forest | High |
| Naive Bayes | High |
| K-Nearest Neighbors | High |
| **Overall System** | **Robust multi-model consensus** |

> Models are evaluated using accuracy, precision, recall, and F1-score. The system uses a consensus/voting strategy across classifiers for final disease prediction.

---

## 🗂️ Dataset

The system is trained on a **symptom–disease dataset** containing:

- **Symptoms** — binary feature encoding of 130+ symptoms per patient record
- **Disease Labels** — 40+ disease categories
- **Medication Data** — mapped medications per disease
- **Precautions** — 4 precautionary steps per disease
- **Diet Recommendations** — suggested dietary habits
- **Workout Suggestions** — physical activity advice per condition

```
datasets/
├── Training.csv              # Symptom–disease training data
├── Testing.csv               # Held-out test set
├── symtoms_df.csv            # Symptom descriptions
├── precautions_df.csv        # Precautionary measures per disease
├── workout_df.csv            # Workout recommendations per disease
├── description.csv           # Disease descriptions
├── medications.csv           # Medication lookup table
└── diets.csv                 # Dietary recommendations
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.8
pandas
numpy
scikit-learn
flask  (or streamlit, if web UI is used)
```

### Installation

```bash
git clone https://github.com/Themahattava/1.-PERSONALIZED-MEDICATION-RECOMMENDATION.git
cd 1.-PERSONALIZED-MEDICATION-RECOMMENDATION
pip install -r requirements.txt
```

### Running the App

```bash
# Flask
python app.py

# OR Streamlit
streamlit run app.py
```

Then open `http://localhost:5000` (Flask) or `http://localhost:8501` (Streamlit) in your browser.

---

## 📁 Project Structure

```
1.-PERSONALIZED-MEDICATION-RECOMMENDATION/
│
├── datasets/
│   ├── Training.csv
│   ├── Testing.csv
│   ├── medications.csv
│   ├── precautions_df.csv
│   ├── diets.csv
│   ├── workout_df.csv
│   └── description.csv
│
├── models/
│   └── svc.pkl               # Saved SVM model (or other classifiers)
│
├── templates/                # HTML templates (if Flask)
│   └── index.html
│
├── static/                   # CSS/JS assets
│
├── app.py                    # Main application entry point
├── medicine_recommendation.ipynb  # Training & exploration notebook
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### 1. Data Preprocessing
- Symptoms are one-hot encoded into a binary feature vector
- Disease labels are label-encoded for classifier compatibility

### 2. Model Training
- Multiple classifiers trained independently on the symptom–disease dataset
- Hyperparameter tuning via cross-validation
- Models serialized with `pickle` for production use

### 3. Prediction Pipeline
- User inputs a list of symptoms
- Input is encoded to match the training feature space
- Each model generates a prediction; final disease is determined by majority vote or highest-confidence model
- Disease label is used to look up medications, precautions, diet, and workout data from CSV tables

### 4. Recommendation Output
For a predicted disease, the system returns:
- ✅ Disease name + description
- 💊 Recommended medications
- ⚠️ Precautions to follow
- 🥗 Dietary suggestions
- 🏃 Workout recommendations

---

## 🖥️ Demo

```
Enter symptoms: itching, skin_rash, nodal_skin_eruptions

Predicted Disease: Fungal infection

Medications     : Antifungal Cream, Fluconazole
Precautions     : Bath twice, use detol or neem in water, avoid scratching, use clean cloths
Diet            : Bath with lukewarm water, apply calamine lotion, eat cooling foods
Workout         : Avoid heavy exercise, light stretching recommended
```

---

## ⚠️ Disclaimer

This system is built for **educational and research purposes only**. It is not a substitute for professional medical diagnosis or advice. Always consult a qualified healthcare professional for medical decisions.

---

## 🙋 Author

**Mahattva** — [@Themahattava](https://github.com/Themahattava)

B.Tech (CSE - Data Science) | Bhilai Institute of Technology, Durg
Research Intern @ IIIT Nagpur (Medical Imaging & Deep Learning)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Symptom–disease dataset sourced from publicly available medical datasets
- Inspired by real-world clinical decision support system (CDSS) research
- Built as part of academic exploration in AI for Healthcare
