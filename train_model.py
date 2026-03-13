import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("dataset/fraud_dataset_95_98_accuracy.csv")

# Features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Save models
pickle.dump(lr, open("models/logistic_model.pkl", "wb"))
pickle.dump(dt, open("models/decision_tree_model.pkl", "wb"))
pickle.dump(rf, open("models/random_forest_model.pkl", "wb"))

print("Models trained and saved successfully!")