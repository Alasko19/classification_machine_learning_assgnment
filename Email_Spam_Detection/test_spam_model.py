import joblib

# STEP 13: LOAD THE SAVED MODEL AND VECTORIZER

model = joblib.load("spam_email_model.pkl")
tfidf = joblib.load("spam_tfidf_vectorizer.pkl")

# new email to test

new_emails = [
    "Subject: Congratulations! You have won a free prize. Click here to claim your reward now.",
    "Subject: Project meeting notes. Hi, please review the attached notes from yesterday's meeting and let me know if you have questions."
]

# convert emails into tf-idf features

new_emails_tfidf = tfidf.transform(new_emails)

# make prediction

predictions = model.predict(new_emails_tfidf)

# display result

print("\nnew email classification results")

for email, prediction in zip(new_emails, predictions):
    print("\nemail:", email)

    if prediction == 1:
        print("Prediction: Spam")
    else:
        print("Prediction: Ham (Not Spam)")