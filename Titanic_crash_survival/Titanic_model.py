import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# STEP 3: LOAD AND EXPLORE THE DATASET

# Loading the Titanic dataset
df = pd.read_csv("data/Titanic.csv")

# Displaying the first five rows
print("\nfirst five row")
print(df.head())

# Displaying the number of rows and columns
print("\ndataset shape")
print(df.shape)

# Displaying all column names
print("\ncolumns names")
print(df.columns.tolist())

# Displaying dataset information
print("\ndataset information")
df.info()

# Checking for missing values
print("\nmissing values")
print(df.isnull().sum())

# Checking the survival distribution
print("\nsurvival distribution")
print(df["survived"].value_counts())

# STEP 4: DATA CLEANING

# checking missing value before cleaning
print("\nmissing value before cleaning")
print(df.isnull().sum())


# filling missing age value with the median
df["age"] = df["age"].fillna(df["age"].median())

# filling the missing embarked values with the mode
df["embarked"] = df["embarked"].fillna(
    df["embarked"].mode()[0]
)

# checking and remove duplicate rows
print("\nduplicate rows:", df.duplicated().sum())


# checking missing value after cleaning
print("\nmissing values after cleaning")
print(df.isnull().sum())

# displaying the clean dataset
print("\nclean dataset")
print(df.head())

print("\ncleaned dataset shape")
print(df.shape)

# STEP 5: DEFINING FEATURES AND TARGET

# selecting the feature (input variables)
x = df.drop("survived", axis=1)

# selecting the target (what we want to predict)
y = df["survived"]

# displaying the features
print("\nfeatures")
print(x.head())

# displaying the target
print("\ntarget")
print(y.head())

# displaying their shape
print("\nfeatures shape:", x.shape)
print("target shape:", y.shape)

# STEP 6: ENCODING CATEGORICAL DATA

# identifying the categorical columns
categorical_columns = [
    "sex",
    "embarked",
    "class",
    "who"
]

# converting the categorical column to numerical columns
x = pd.get_dummies(
    x,
    columns=categorical_columns,
    drop_first=True
)

# converting the alone column from bolean to integer
x["alone"] = x["alone"].astype(int)

# displaying the encoded features
print("\nencoded features")
print(x.head())

# checking the column names
print("\nencoded column names")
print(x.columns.tolist())

# checking the data types
print("\ndata types")
print(x.dtypes)

# checking the new dataset dimensions
print("\nencoded feature shape:", x.shape)


# STEP 7: SPLITING THE DATASET
# spliting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# displaying the shapes of the datasets
print("\ntraining and testing data")

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# checking survival distribution in training data
print("\ntraining survival distribution")
print(y_train.value_counts())

# checking survival distribution in testing data
print("\ntesting survival distribution")
print(x_test.value_counts())


# STEP 8: BUILDING AND TRAINING THE MODEL

# creating the logistic regression model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# training the model
model.fit(x_train, y_train)

print("\nmodel training completed successfully!")

# STEP 9: MAKING PREDICTION

y_pred = model.predict(x_test)

print("\nmodel prediction")
print(y_pred)

# STEP 10: EVALUATING MODEL PERFORMANCE
accuracy = accuracy_score(y_test, y_pred)

print("\nmodel accuracy:")
print(accuracy)

print("\nconfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nclassification report")
print(classification_report(y_test, y_pred))

# STEP 11: VISUALIZING THE CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Did Not Survive", "Survived"],
    yticklabels=["Did Not Survive", "Survived"]
)

plt.xlabel("Predicted Outcome")
plt.ylabel("Actual Outcome")
plt.title("Titanic Survive Prediction - Confusion Matrix")

plt.tight_layout()
plt.show()

# STEP 12: SAVING THE TRAINED MODEL

joblib.dump(model, "titanic_survival_model.pkl")
print("\nmodel saved successfully")

