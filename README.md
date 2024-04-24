# au810021102503_thangapandi
Heart Disease Prediction - You are tasked to perform Heart Disease Prediction Using Logistic Regression. The World Health Organization has estimated that four out of five cardiovascular disease (CVD) deaths are due to heart attacks. This whole research intends to pinpoint the ratio of patients who have a good chance of being affected by CVD 
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("heart_disease_data.csv")

# Data preprocessing
# Handle missing values, outliers, encode categorical variables, normalize/standardize features

# Split the dataset into features and target variable
X = data.drop("target", axis=1)
y = data["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Plot ROC curve and calculate AUC
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print("AUC Score:", roc_auc_score(y_test, y_prob))

# Interpret the results
# Analyze coefficients, identify influential factors

# Predicting risk for new individuals
# Input new feature values into the trained model
