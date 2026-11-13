import joblib
import pandas as pd

# loading the trained model
model = joblib.load("titanic_survival_model.pkl")

print("model loaded successfully")


# creating a sample passenger information
sample_passenger = pd.DataFrame([{
    "age": 25,
    "sibsp": 0,
    "parch": 0,
    "fare": 50.0,
    "sex": "female",
    "embarked": "S",
    "class": "Third",
    "who": "woman",
    "alone": True
}])

# encoding categorical features
sample_passenger = pd.get_dummies(
    sample_passenger,
    columns=["sex", "embarked", "class", "who"],
    drop_first=True
)

sample_passenger["alone"] = sample_passenger["alone"].astype(int)

# matching the sample columns to the training columns
sample_passenger = sample_passenger.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)

# making a prediction
prediction = model.predict(sample_passenger)

print("\npassenger survival prediction:")

if prediction[0] == 1:
    print("the passenger is predicted to survive.")
else:
    print("the passeneger is predicted not to survive.")