import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('clean_data.csv')

# Split the data into features and target variable
X = data.drop('isFraud', axis=1)
y = data['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# hyperparameters
param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': ['balanced', None]
}

rf_model = RandomForestClassifier(min_samples_leaf=3,
                                  bootstrap=False,
                                  min_samples_split=5,
                                  class_weight='balanced',
                                  random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=5,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='f1'
)

random_search.fit(X_train, y_train)

# Get the best estimator
best_rf_model = random_search.best_estimator_

# Predictions using the best found parameters
best_rf_predictions = best_rf_model.predict(X_test)

rf_model.fit(X_train, y_train)
baseline_rf_predictions = rf_model.predict(X_test)

print("\nRandom Forest Classifier Results:")
print(classification_report(y_test, baseline_rf_predictions, zero_division=0))
print("Accuracy:", accuracy_score(y_test, baseline_rf_predictions))

print("\nRandom Forest Classifier Results after Hyperparameter Tuning:")
print(classification_report(y_test, best_rf_predictions, zero_division=0))
print("Accuracy:", accuracy_score(y_test, best_rf_predictions))