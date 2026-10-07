# NLP Text Classification using Machine Learning

An NLP-based text classification project that uses **TF-IDF Vectorization** and **Logistic Regression** to classify text documents into four news categories.

The model achieves **89.1% test accuracy** on the selected subset of the 20 Newsgroups dataset.

## Project Overview

Text classification is a common Natural Language Processing (NLP) task where a machine learning model learns to assign a category to a given text document.

In this project, the text is transformed into numerical features using **TF-IDF**, and a **Logistic Regression** classifier is trained to predict the category.

### Classification Categories

- `comp.graphics`
- `rec.sport.hockey`
- `sci.med`
- `sci.space`

## Results

| Metric | Score |
|---|---:|
| Test Accuracy | **89.1%** |
| Macro Average F1-score | **0.89** |
| Weighted Average F1-score | **0.89** |

### Class-wise Performance

| Category | Precision | Recall | F1-score |
|---|---:|---:|---:|
| comp.graphics | 0.90 | 0.89 | 0.90 |
| rec.sport.hockey | 0.98 | 0.92 | 0.95 |
| sci.med | 0.90 | 0.86 | 0.88 |
| sci.space | 0.80 | 0.89 | 0.85 |

## Confusion Matrix

The confusion matrix shows how the model performs across the four categories.

The main diagonal represents correctly classified documents, while the off-diagonal values represent classification errors.

<img width="513" height="437" alt="image" src="https://github.com/user-attachments/assets/137df9ea-71ba-4ecd-9e2f-3444d1beb4bb" />


## Dataset

This project uses the **20 Newsgroups** dataset available through Scikit-learn.

Only four categories are selected:

- `sci.med`
- `sci.space`
- `rec.sport.hockey`
- `comp.graphics`

Headers, footers, and quoted text are removed to reduce potential information leakage.

The dataset used in the experiment contains:

- **2,371 training documents**
- **1,578 test documents**

## Machine Learning Pipeline

The project follows this workflow:

```text
Text Documents
      ↓
Dataset Loading
      ↓
Text Cleaning
(headers, footers, quotes removed)
      ↓
TF-IDF Vectorization
      ↓
Logistic Regression
      ↓
Prediction
      ↓
Evaluation
      ↓
Confusion Matrix
```

## Technologies Used

- Python
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- Joblib
- TF-IDF
- Logistic Regression

## Key Techniques

### 1. TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical features.

The implementation uses:

- Lowercase conversion
- English stop-word removal
- Unigrams and bigrams
- Maximum 30,000 features

```python
TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=30000
)
```

### 2. Logistic Regression

Logistic Regression is used as the classification algorithm.

```python
LogisticRegression(max_iter=1000)
```

### 3. Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Example Predictions

The trained model correctly identifies example sentences such as:

```text
"NASA launched a new satellite to orbit Mars"
→ sci.space

"The goalie made 30 saves in the playoff game"
→ rec.sport.hockey
```

## Project Structure

```text
nlp-text-classification/
│
├── nlp_text_classification.py
├── README.md
├── requirements.txt
└── nlp_project/
    ├── confusion_matrix.png
    └── text_classifier.joblib
```

> `nlp_project/` is generated when the Python script is executed.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/nlp-text-classification.git
cd nlp-text-classification
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Project

Run:

```bash
python nlp_text_classification.py
```

The script will:

1. Download/load the selected 20 Newsgroups dataset.
2. Train the TF-IDF + Logistic Regression pipeline.
3. Evaluate the model.
4. Display the classification report.
5. Generate the confusion matrix.
6. Save the trained model as `text_classifier.joblib`.
7. Run example predictions.

## Model Output

The trained pipeline is saved using Joblib:

```text
nlp_project/text_classifier.joblib
```

This allows the trained model to be reused later without retraining.

## What I Learned

Through this project, I practiced:

- Natural Language Processing fundamentals
- Text preprocessing
- TF-IDF feature extraction
- N-gram feature engineering
- Logistic Regression for text classification
- Model evaluation
- Precision, Recall and F1-score
- Confusion matrix analysis
- Saving machine learning models with Joblib
- Building an end-to-end ML pipeline with Scikit-learn

## Future Improvements

Possible improvements include:

- Comparing Logistic Regression with Naive Bayes and Linear SVM
- Hyperparameter tuning
- Testing additional Newsgroups categories
- Improving text preprocessing
- Adding a web interface using Streamlit
- Deploying the classifier as an API
- Adding interactive prediction functionality

## Author

**AMIT SHAHARE**

MCA Student | Data Analyst | Machine Learning & NLP Enthusiast

---

If you found this project useful, consider giving the repository a ⭐.
