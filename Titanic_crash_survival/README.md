Titanic Survival Prediction

Project Overview

This project uses Machine Learning classification to predict whether a passenger survived the Titanic disaster based on passenger information.

Dataset

The dataset contains 891 passenger records and 10 columns, including age, sex, passenger class, fare, and survival status.

Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

Machine Learning Algorithm

Logistic Regression was used to build the binary classification model.

Project Workflow

1. Dataset loading and exploration
2. Data cleaning and preprocessing
3. Handling missing values
4. Encoding categorical variables
5. Splitting data into training and testing sets
6. Model training
7. Making predictions
8. Model evaluation
9. Confusion matrix visualization
10. Saving and loading the trained model

Model Performance

* Accuracy: 83.80%
* Precision (Survived): 0.81
* Recall (Survived): 0.75
* F1-score (Survived): 0.78

Files

* Titanic_model.py – Model development and evaluation.
* test_titanic_model.py – Loading the saved model and testing predictions.
* titanic_survival_model.pkl – Saved trained model.
* data/ – Dataset folder.

Dataset Source

The dataset used in this project was obtained from Kaggle:

* Dataset name: Titanic Dataset
* Platform: Kaggle
* Dataset link: [https://www.kaggle.com/datasets/ibrahimelsayed182/titanic-dataset]

Note: The dataset CSV file is not included in this GitHub repository. Please refer to the original Kaggle source for downloading the data.

Conclusion

The Logistic Regression classifier achieved 83.80% accuracy on the test dataset. This project demonstrates a complete beginner-level machine learning classification workflow.