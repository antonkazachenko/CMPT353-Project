import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


data = pd.read_csv('C:/Users/kiril/Desktop/CMPT/353/final project/CMPT353-Project/clean_data.csv')

# Split the data into features and target variable
X = data.drop('isFraud', axis=1)
y = data['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# hyperparameters
param_grid = {
    'n_estimators': [100, 200]
    # 'max_depth': [None, 10],
    # 'min_samples_split': [2, 5],
    # 'class_weight': ['balanced', None]
}

rf_model = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf_model, param_distributions=param_grid, cv=5, random_state=42)
random_search.fit(X_train, y_train)
best_rf_model = random_search.best_estimator_
best_rf_predictions = best_rf_model.predict(X_test)

print("Best Parameters:", random_search.best_params_)
print("Random Forest Classifier Results with Best Parameters:")
print(classification_report(y_test, best_rf_predictions, zero_division=0))
print("Accuracy:", accuracy_score(y_test, best_rf_predictions))