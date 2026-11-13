import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib



#STEP 2: 
# loading the dataset
df = pd.read_csv("data/spam_ham_dataset.csv")

# displaying the first five record
print("\nfirst five email record")
print(df.head())

# checking the dimension
print("\ndataset shape")
print(df.shape)

# checking column names
print("\ndataset columns")
print(df.columns.tolist())

# checking dataset information
print("\ndataset information")
df.info()

# checking missing values
print("\nmissing value")
print(df.isnull().sum())

# checking spam and ham distribution
print("\nemail label distribution")
print(df["label"].value_counts())


# STEP 3: CHECKING EMAIL DISTRIBUTION AND DUPLICATE RECORD

# checking duplicate email record
print("\nduplicate email record")
print(df.duplicated(subset=["text"]).sum())

# checking numerical label distribution
print("\nnumerical label distribution")
print(df["label_num"].value_counts())

# STEP 4: CLEANING THE DATA

# removing unecessary column
df = df.drop(columns=["Unnamed: 0"])

# checking duplicate email with different label
conflicting_labels = (
    df.groupby("text")["label"].nunique() > 1
).sum()

print("\nduplicate email with conflicting labels:")
print(conflicting_labels)

# removing duplicate email record
df = df.drop_duplicates(subset=["text"])

# resetting dataset index
df = df.reset_index(drop=True)

# checking clean dataset
print("\ncleaned dataset")
print(df.head())

print("\ncleaned dataset shape")
print(df.shape)

print("\nmissing value after cleaning")
print(df.isnull().sum())

print("\nemail distribution after cleaning")
print(df["label"].value_counts())

print("\nduplicates after cleaning")
print(df.duplicated(subset=["text"]).sum())

# STEP 5: SEPARATING THE FEATURES AND TARGET
# selecting email text as our features
x = df["text"]

# selecting numerical label as our target
y = df["label_num"]

# displaying features
print("\nemail text features")
print(x.head())

# displaying target
print("\nemail classification target")
print(y.head())

# checking dataset dimension
print("\nfeatures shape:", x.shape)
print("target shape:", y.shape)

#checking target distribution
print("\ntarget distribution")
print(y.value_counts())

# STEP 6: SPLITTING THE DATASET
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
) 

# displaying dataset shape
print("\ntraining and testing dataset")

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# checking training email distribution
print("\ntraining email distribution")
print(y_train.value_counts())

# checking testing email distribution
print("\ntesting email distribution")
print(y_test.value_counts())

# STEP 7: CONVERTING EMAIL TEXT INTO NUMERICAL FEATURES

# creating the TF-IDF vectorizer
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

# learning vocabulary from training email
x_train_tfidf = tfidf.fit_transform(x_train)

# transforming testing email using the same vocabulary
x_test_tfidf = tfidf.transform(x_test)

# displaying transform data
print("\ntf-idf transformation completed")

print("training data shape:", x_train_tfidf.shape)
print("testing data shape:", x_test_tfidf.shape)

print("\nsample tf-idf features")
print(tfidf.get_feature_names_out()[:20])

# STEP 8: BUILDING AND TRAINING THE CLASSIFICATION MODEL

#creating the Multinomial Naive Bayes model
model = MultinomialNB()

# training the model
model.fit(x_train_tfidf, y_train)

print("\nspam classification model trained successfully!")


# STEP 9: MAKING SPAM EMAIL PREDICTION

y_pred = model.predict(x_test_tfidf)

print("\nspam email predictions")
print(y_pred)

# STEP 10: EVALUATE MODEL PERFORMANCE

accuracy = accuracy_score(y_test, y_pred)

print("\nmodel accuracy:")
print(accuracy)

print("\nconfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nclassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham (Not Spam)", "Spam"]
    )
)

# STEP 11: VISUALIZING THE CONFUSION EMAIL

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham (Not Spam)", "Spam"],
    yticklabels=["Ham (Not Spam)", "Spam"]
)

plt.xlabel("Predicted Email")
plt.ylabel("Actual Email")
plt.title("Spam Email Detection - confusion matrix")

plt.tight_layout()
plt.show()

# STEP 12: SAVING THE TRAINED MODEL AND VECTORIZER
joblib.dump(model, "spam_email_model.pkl")
joblib.dump(tfidf, "spam_tfidf_vectorizer.pkl")

print("\nmodel and vectorizer saved successfully")