# Student Performance Prediction

## Overview

This project predicts student academic performance based on features such as previous grades, study time, number of past failures, and absences. The goal is to identify students who may need additional support to succeed and enable early intervention strategies. Several machine learning models, including Ridge Regression, Random Forest, and Gradient Boosting, are employed to perform predictions. Hyperparameter tuning and feature selection are also integrated to optimize the performance of the models.

## Features

- **Data Preprocessing:** Handles normalization and scaling of input data for better model performance.
- **Feature Selection:** Selects the most relevant features using SelectKBest for improved accuracy.
- **Model Training:** Uses Ridge Regression, Random Forest, and Gradient Boosting for training and prediction.
- **Hyperparameter Tuning:** Employs GridSearchCV to optimize hyperparameters for each model.
- **Cross-Validation:** Implements k-fold cross-validation to ensure robustness and prevent overfitting.
- **Model Evaluation:** Evaluates models based on Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- **User Input Interface:** Allows users to input new data for predictions using the best-performing model.

## Dataset

The dataset used in this project is related to the academic performance of students in secondary school. It includes features like previous grades (G1 and G2), study time, failures, and absences.

- **File:** `student_data.csv`
- **Features Used:**
  - `G1` (First period grade)
  - `G2` (Second period grade)
  - `studytime` (Study time per week)
  - `failures` (Number of past failures)
  - `absences` (Number of absences)
  - `G3` (Final grade - target variable)

## Dependencies

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- 
## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/rajatsharma05/grade_predict.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd grade_predict
   ```

3. **Prepare the Dataset:**

   Ensure that `student_data.csv` is available in the project directory.

4. **Run the Script:**

   Execute the script to train the models, evaluate performance, and predict student performance.

   ```bash
   python predict.py
   ```

5. **Interactive Prediction:**

   After model training, the script will ask for user input to predict a student's final grade (G3) based on new data.

## Model Comparison

The following models are used in this project:

- **Ridge Regression:** A linear regression model with L2 regularization, optimized through RidgeCV.
- **Random Forest Regressor:** An ensemble method that constructs multiple decision trees for accurate predictions.
- **Gradient Boosting Regressor:** Another ensemble technique that builds models sequentially, with each correcting the errors of the previous model.

Each model's performance is evaluated and the best model is selected for making predictions.

## Visualization

The script also generates a scatter plot showing the actual vs. predicted values for the best-performing model to visualize how well the model performed.

## Contributions

Contributions are welcome! Please feel free to submit issues or pull requests to enhance this project.

## Acknowledgments

This project was inspired by the growing need for predictive analytics in education and is based on publicly available student performance datasets.
dataset was taken from https://archive.ics.uci.edu/dataset/320/student+performance

---

Thank you for checking out the project! If you have any questions or suggestions, feel free to reach out.
