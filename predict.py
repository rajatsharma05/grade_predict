import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv("student_data.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Define features and target
predict = "G3"
X = data.drop(columns=[predict])
y = data[predict]

# Normalize/Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter Tuning
ridge_params = {'alphas': [0.1, 1.0, 10.0]}
ridge_grid = GridSearchCV(RidgeCV(), ridge_params, cv=5)
ridge_grid.fit(X_scaled, y)
best_alpha = ridge_grid.best_estimator_.alpha_

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1)

# Create and train the model with Ridge Regression
ridge = RidgeCV(alphas=[best_alpha], cv=5)
ridge.fit(x_train, y_train)

# Evaluate the model
acc_train = ridge.score(x_train, y_train)
acc_test = ridge.score(x_test, y_test)
print("Train Accuracy (Ridge):", acc_train)
print("Test Accuracy (Ridge):", acc_test)

# Cross-validation
cv_scores = cross_val_score(ridge, X_scaled, y, cv=5)
print("Cross-Validation Mean Accuracy (Ridge):", np.mean(cv_scores))

# Feature Selection
# uses selectkbest
selector = SelectKBest(score_func=f_regression, k=3)  # Select 3 best features
X_selected = selector.fit_transform(X_scaled, y)

# Split selected features into train and test sets
x_train_sel, x_test_sel, _, _ = train_test_split(X_selected, y, test_size=0.1)

# Create and train the model with selected features
ridge_sel = RidgeCV(alphas=[best_alpha], cv=5)
ridge_sel.fit(x_train_sel, y_train)

# Evaluate the model with selected features
acc_train_sel = ridge_sel.score(x_train_sel, y_train)
acc_test_sel = ridge_sel.score(x_test_sel, y_test)
print("Train Accuracy (Ridge with Selected Features):", acc_train_sel)
print("Test Accuracy (Ridge with Selected Features):", acc_test_sel)

# Ensemble Methods aka combination methods (basically combines multiple methods in a head to head to figure out best accuracy %.)
# Random Forest
rf_params = {'n_estimators': [50, 100, 200]}
rf_grid = GridSearchCV(RandomForestRegressor(), rf_params, cv=5)
rf_grid.fit(X_scaled, y)
best_rf_params = rf_grid.best_params_
rf = RandomForestRegressor(**best_rf_params)
rf.fit(x_train, y_train)
rf_acc_train = rf.score(x_train, y_train)
rf_acc_test = rf.score(x_test, y_test)
print("Train Accuracy (Random Forest):", rf_acc_train)
print("Test Accuracy (Random Forest):", rf_acc_test)

# Gradient Boosting
gb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
gb_grid = GridSearchCV(GradientBoostingRegressor(), gb_params, cv=5)
gb_grid.fit(X_scaled, y)
best_gb_params = gb_grid.best_params_
gb = GradientBoostingRegressor(**best_gb_params)
gb.fit(x_train, y_train)
gb_acc_train = gb.score(x_train, y_train)
gb_acc_test = gb.score(x_test, y_test)
print("Train Accuracy (Gradient Boosting):", gb_acc_train)
print("Test Accuracy (Gradient Boosting):", gb_acc_test)

# Model Evaluation and Prediction
models = [("Ridge", ridge), ("Ridge with Selected Features", ridge_sel), ("Random Forest", rf), ("Gradient Boosting", gb)]
best_model_name, best_model = max(models, key=lambda x: x[1].score(x_test if x[0] != "Ridge with Selected Features" else x_test_sel, y_test))
print("Best Model:", best_model_name)

for name, model in models:
    if name == "Ridge with Selected Features":
        predictions = model.predict(x_test_sel)
    else:
        predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model: {name}")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)

# Plot actual vs. predicted values for the best model (for Debugging and visualization wont be in final)
best_predictions = best_model.predict(x_test if best_model_name != "Ridge with Selected Features" else x_test_sel)
plt.scatter(y_test, best_predictions)
plt.xlabel("Actual")
plt.ylabel("Predicted") 
plt.title("Actual vs. Predicted Values (Best Model)")
plt.show()

# Take input from the user for prediction parameters
print("\nPlease provide the following details for prediction:")
G1 = float(input("Enter G1 (first period grade): "))
G2 = float(input("Enter G2 (second period grade): "))
studytime = float(input("Enter study time (hours per week): "))
failures = float(input("Enter number of past failures: "))
absences = float(input("Enter number of absences: "))


input_data = np.array([[G1, G2, studytime, failures, absences]])
input_data_scaled = scaler.transform(input_data)

# Use the trained model to predict the outcome
if best_model_name == "Ridge with Selected Features":
    prediction = best_model.predict(selector.transform(input_data_scaled))
else:
    prediction = best_model.predict(input_data_scaled)

print("Predicted Grade (G3):", prediction[0])