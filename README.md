# Twitter-Sentiment-Analysis
# Twitter Sentiment Analysis using Machine Learning

This project performs sentiment analysis on tweets using the [Sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140). The goal is to classify tweets as **positive** or **negative** using natural language processing (NLP) techniques and a logistic regression model.

## 📁 Dataset

- **Source**: [Sentiment140 - Kaggle](https://www.kaggle.com/kazanova/sentiment140)
- **Size**: 1.6 million tweets
- **Columns**:
  - `target`: Sentiment label (0 = negative, 4 = positive)
  - `id`: Tweet ID
  - `date`: Timestamp
  - `flag`: Query flag (not used)
  - `user`: Username
  - `text`: Tweet text

> 🔄 For binary classification: label `4` is mapped to `1` (positive)

## 🔧 Project Workflow

1. **Data Loading & Cleaning**
   - Read dataset and assign column names
   - Remove special characters, lowercase text
   - Tokenize and remove stopwords
   - Apply stemming using NLTK’s `PorterStemmer`

2. **Feature Engineering**
   - TF-IDF Vectorization to convert text to numerical features

3. **Model Building**
   - Train-test split (80-20)
   - Logistic Regression model using `sklearn`
   - Accuracy score: ~77.8%

4. **Model Persistence**
   - Trained model saved using `pickle`

5. **Prediction**
   - Load saved model to predict sentiment on new tweet samples

## 🧪 Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- scikit-learn
- TF-IDF Vectorizer
- Pickle
- Jupyter Notebook / Google Colab

## 📊 Results

- **Training Accuracy**: 77.8%
- **Testing Accuracy**: 77.8%
- The model generalizes well and is suitable for basic sentiment classification.




## 📌 Overview

- Preprocess tweets (cleaning, stemming, stopword removal)
- Convert text into vectors using **TF-IDF**
- Train a **Logistic Regression** model
- Achieve around **77.8% accuracy**
- Save and load model with **pickle**
- Make sentiment predictions on new tweet samples

---

## 📁 Dataset

- **Source**: [Sentiment140 on Kaggle](https://www.kaggle.com/kazanova/sentiment140)
- **Format**: CSV, 1.6 million tweets
- **Target Mapping**:
  - `0` → Negative
  - `4` → Positive → converted to `1`

---

## ⚙️ Environment & Tools

- **Platform**: Google Colab  
- **Libraries Used**:
  - `pandas`, `numpy`
  - `nltk` (for stopwords and stemming)
  - `scikit-learn` (modeling and evaluation)
  - `pickle` (model persistence)

---

## 📌 How to Run on Google Colab

1. **Open the notebook in Colab**
   > You can upload the `.ipynb` file or clone the repo into your Google Drive.

2. **Upload Kaggle API Key**
   - Upload your `kaggle.json` using the file upload interface
   - Set permissions and download the dataset via Colab:
     ```python
     !pip install kaggle
     !mkdir -p ~/.kaggle
     !cp kaggle.json ~/.kaggle/
     !chmod 600 ~/.kaggle/kaggle.json
     !kaggle datasets download -d kazanova/sentiment140
     ```

3. **Extract and Load Data**
   - Extract the ZIP and load the CSV into a DataFrame
   - Preprocess text data

4. **Train the Model**
   - Use `TfidfVectorizer`
   - Train a `LogisticRegression` model
   - Evaluate accuracy on train and test data

5. **Save and Load Model**
   ```python
   import pickle
   pickle.dump(model, open('trained_model.sav', 'wb'))
   model = pickle.load(open('trained_model.sav', 'rb'))
