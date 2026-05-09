# Fake Review Detection and Sentiment Analysis

A Machine Learning based web application that detects whether a product review is **Fake or Real** using NLP and Machine Learning techniques.

The project is built using:
- Python
- Streamlit
- Scikit-learn
- NLP
- Machine Learning

---

# Project Overview

Fake reviews are a major problem in e-commerce platforms because they can mislead customers and affect product trustworthiness.

This project analyzes user reviews and predicts whether the review is:
- Fake
- Real

The application uses Natural Language Processing (NLP) and Machine Learning models trained on review datasets.

---

# Features

- Fake Review Detection
- NLP based text preprocessing
- Machine Learning prediction
- Streamlit Web Interface
- Real-time review analysis
- Easy to use UI

---

# Technologies Used

## Programming Language
- Python

## Libraries & Frameworks
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Joblib

## Machine Learning
- TF-IDF Vectorization
- Logistic Regression / Naive Bayes

---

# Project Structure

```bash
Fake-Review-Detection/
│
├── app.py
├── fake_review_model.joblib
├── vectorizer.joblib
├── requirements.txt
└── README.md
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/fake-review-detection.git
```

## Move into the Project Directory

```bash
cd fake-review-detection
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Application

```bash
streamlit run app.py
```

---

# How the Project Works

## Step 1: User Enters a Review
The user inputs a product review in the Streamlit application.

## Step 2: Text Preprocessing
The review text is cleaned using NLP techniques:
- Lowercasing
- Removing punctuation
- Removing stopwords
- Tokenization

## Step 3: TF-IDF Vectorization
The processed text is converted into numerical vectors using TF-IDF.

## Step 4: Prediction
The trained Machine Learning model predicts whether the review is:
- Fake
- Real

## Step 5: Result Display
The prediction result is displayed on the Streamlit interface.

---

# Sample Input & Output

| Review | Prediction |
|--------|-------------|
| This product is amazing and worth buying | Real |
| Best product ever!!!!! Buy now!!! | Fake |

---

# Requirements

```txt
streamlit
pandas
numpy
scikit-learn
nltk
joblib
```

---

# Future Enhancements

- Add Sentiment Analysis
- Scrape reviews directly from Amazon/Flipkart
- Deploy on cloud platforms
- Improve model accuracy
- Add graphical dashboard
- Multi-language support

---

# Advantages

- Helps identify misleading reviews
- Improves customer trust
- Fast and automatic prediction
- Easy to use interface

---

# Challenges Faced

- Dataset preprocessing
- Handling noisy text
- Fake review classification accuracy
- NLP model optimization

---

# Learning Outcomes

Through this project, we learned:
- Machine Learning workflow
- NLP preprocessing techniques
- Streamlit development
- Model deployment
- Real-world AI applications

---

# Conclusion

This project demonstrates how Machine Learning and NLP techniques can be used to detect fake product reviews effectively. The system helps users make better purchasing decisions by identifying misleading reviews.

---

# Author

Purvi S Kiran

---

# License

This project is for educational purposes only.
