# Salary Prediction Notebook

This repository contains a Jupyter Notebook (`Salary.ipynb`) that predicts salaries using various regression models. The notebook processes a dataset (`Salary Data.csv`) from Kaggle, conducts exploratory data analysis, and applies machine learning techniques to build and evaluate models for predicting salaries.

## Features

1. **Data Loading and Cleaning**:

   - Loads the dataset from a CSV file.
   - Handles missing data through replacement and imputation.

2. **Exploratory Data Analysis**:

   - Analyzes categorical variables like `Gender` and `Education Level`.
   - Inspects numerical variables like `Age` and `Years of Experience`.
   - Generates distribution plots and summary statistics for deeper insights.

3. **Feature Engineering**:

   - Separates the target variable (`Salary`) from predictors.
   - Encodes categorical features and scales numerical features using `OneHotEncoder` and `StandardScaler`.
   - Splits the dataset into training and test sets to ensure robust model evaluation.

4. **Modeling**:

   - Implements multiple regression models:
     - Linear Regression
     - Ridge Regression
     - Lasso Regression
     - Decision Tree
     - K-Neighbors Regressor
     - Random Forest
   - Performs hyperparameter tuning for Ridge and Lasso models to optimize performance.
   - Evaluates models using metrics:
     - R² (Coefficient of Determination)
     - MAE (Mean Absolute Error)
     - RMSE (Root Mean Squared Error)

5. **Visualization**:

   - Plots Actual vs. Predicted values to assess model accuracy visually.
   - Uses regression plots to showcase trends and model fits.
   - Highlights feature importance for tree-based models.

## Prerequisites

- Python 3.8+
- Required libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

## Installation

1. Clone this repository:

   ```bash
   git clone <repository_url>
   ```

2. Navigate to the project directory:

   ```bash
   cd <repository_name>
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the dataset (`Salary Data.csv`) is in the same directory as the notebook.

## Usage

1. Open the notebook in Jupyter:

   ```bash
   jupyter notebook Salary.ipynb
   ```

2. Run the cells step-by-step to:

   - Load and preprocess the data.
   - Train regression models.
   - Evaluate and visualize the results.

3. Inspect the final DataFrame containing actual, predicted, and difference values.

## Results

- The notebook compares multiple regression models, showing performance variations.
- Ridge Regression achieves the best performance with an R² score above 0.90, indicating strong predictive power.
- Visualizations provide a clear comparison between actual and predicted values, highlighting model efficiency.
- Key findings include:
  - Strong correlation between `Years of Experience` and `Salary`.
  - Ridge and Lasso Regression models effectively reduce overfitting compared to Linear Regression.

## Conclusion

This project demonstrates the application of machine learning regression techniques to predict salaries. The Ridge Regression model emerged as the most effective, offering a balance of simplicity and accuracy. The notebook provides a structured approach to data preprocessing, model training, and evaluation, making it a valuable resource for similar predictive tasks.

Future enhancements could include:
- Incorporating additional features for better predictions.
- Experimenting with advanced models like Gradient Boosting or Neural Networks.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions, feel free to submit an issue or pull request.

