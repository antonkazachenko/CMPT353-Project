import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('C:/Users/kiril/Desktop/CMPT/353/final project/CMPT353-Project/clean_data.csv')

# Split the data into features and target variable
X = data.drop('isFraud', axis=1)
y = data['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(random_state=42)
gbm_model = GradientBoostingClassifier(random_state=42)

# Train the Random Forest model
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("\nRandom Forest Classifier Results:")
print(classification_report(y_test, rf_predictions, zero_division=0))
print("Accuracy:", accuracy_score(y_test, rf_predictions))

# Train the SVM model
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
print("\nSVM Classifier Results:")
print(classification_report(y_test, svm_predictions, zero_division=0))
print("Accuracy:", accuracy_score(y_test, svm_predictions))

# Train the GBM model
gbm_model.fit(X_train, y_train)
gbm_predictions = gbm_model.predict(X_test)
print("\nGradient Boosting Classifier Results:")
print(classification_report(y_test, gbm_predictions, zero_division=0))
print("Accuracy:", accuracy_score(y_test, gbm_predictions))
