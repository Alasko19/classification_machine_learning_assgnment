# Spam Email Detection Using Machine Learning

# Project Overview
This project uses machine learning to classify emails as either Spam or Ham (Not Spam).

The goal is to build a model that can automatically identify unwanted emails.

# Dataset
The dataset was obtained from Kaggle.

- Original records: 5,171
- Records after cleaning: 4,993
- Ham emails: 3,531
- Spam emails: 1,462

# Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

# Machine Learning Algorithm
Multinomial Naive Bayes was used for classification.

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert email text into numerical features.

# Project Workflow
1. Loaded and explored the dataset.
2. Cleaned the dataset and removed duplicate emails.
3. Separated features and target labels.
4. Split the dataset into training and testing sets.
5. Converted email text using TF-IDF.
6. Trained the Multinomial Naive Bayes model.
7. Evaluated model performance.
8. Saved the trained model and vectorizer.
9. Tested predictions on new emails.

# Model Performance

- Accuracy: 94.89%
- Spam Precision: 88%
- Spam Recall: 96%
- Spam F1-Score: 92%

# Project Files
- `spam_classifier.py` - Main model training and evaluation script.
- `test_spam_model.py` - Tests the saved model with new emails.
- `data/spam_ham_dataset.csv` - Dataset.
- `spam_email_model.pkl` - Saved trained model.
- `spam_tfidf_vectorizer.pkl` - Saved TF-IDF vectorizer.

# Dataset Source

The dataset used in this project was obtained from Kaggle:

* Dataset name: Spam/Ham Email Classification Dataset
* Platform: Kaggle
* Dataset link: [https://www.kaggle.com/datasets/venky73/spam-mails-dataset]

Note: The dataset CSV file is not included in this GitHub repository. Please refer to the original Kaggle source for downloading the data.

## Conclusion
The model achieved approximately 94.89% accuracy on the test dataset and successfully classified sample new emails as Spam or Ham.