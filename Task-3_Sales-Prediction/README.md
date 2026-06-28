# Sales Prediction Using Python

## Project Overview

Sales prediction is the process of forecasting future product sales based on advertising expenditures. In this project, a **Machine Learning model** is developed using the **Linear Regression algorithm** to predict sales based on advertising budgets spent on **TV, Radio, and Newspaper** platforms.

This project demonstrates how businesses can use data analysis and machine learning techniques to optimize advertising strategies and improve sales forecasting.

## Objective

The main objectives of this project are:

   * Analyze advertising expenditure data.
   * Identify the relationship between advertising platforms and sales.
   * Build a Machine Learning model for sales prediction.
   * Evaluate model performance using regression metrics.
   * Predict future sales based on advertising budgets.

## Dataset Information

The dataset contains **200 records** and **4 features**.

| Feature   | Description                           |
| --------- | ------------------------------------- |
| TV        | Advertising budget spent on TV        |
| Radio     | Advertising budget spent on Radio     |
| Newspaper | Advertising budget spent on Newspaper |
| Sales     | Product sales                         |

## Technologies Used

   * Python
   * Matplotlib
   * Scikit-learn
   * Joblib

## Libraries Used

```python
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
```

## Data Analysis

The following data analysis techniques performed:

   * Dataset inspection
   * Statistical summary
   * Missing value analysis
   * Correlation analysis
   * Data visualization

## Data Visualization

The following visualizations created:

### 1. TV Advertising vs Sales

A scatter plot showing the relationship between TV advertising expenditure and sales.

  **TV Advertising vs Sales GRAPH**

![alt text](<TV Advertising vs Sales.png>)

### 2. Actual vs Predicted Sales (Scatter Plot)

A scatter plot comparing actual sales values with predicted sales values.

 **Actual vs Predicted Sales(Scatter Plot) Graph**

![alt text](<Actual vs Predicted Sales.png>)

### 3. Actual vs Predicted Sales (Line Plot)

A line graph comparing actual and predicted sales performance.

 **Actual vs Predicted Sales(Line Plot) Graph**

![alt text](<Actual vs Predicted Sales (1).png>)

## Machine Learning Model

In this project we use the **Linear Regression** algorithm.

### Steps Performed:

1. Data Loading
2. Data Exploration
3. Data Visualization
4. Feature Selection
5. Train-Test Split
6. Model Training
7. Model Prediction
8. Model Evaluation
9. Model Saving

## Model Performance

| Metric                    | Value  |
| ------------------------- | ------ |
| Mean Absolute Error (MAE) | 1.2748 |
| Mean Squared Error (MSE)  | 2.9078 |
| R² Score                  | 0.9059 |
| Model Accuracy            | 90.59% |

## Feature Importance

| Feature   | Coefficient |
| --------- | ----------- |
| TV        | 0.054509    |
| Radio     | 0.100945    |
| Newspaper | 0.004337    |

### Model Intercept
```text
4.714126402214127
```

## Sample Prediction

### Input Data
```text
TV = 230.1
Radio = 37.8
Newspaper = 69.2
```
### Predicted Sales
```text
21.37
```
## Model Saving

The trained model is saved using **Joblib**:

This model can be reused later without retraining.

## Prediction Results

The actual and predicted sales values are stored in:-
 Sales_prediction_result.csv

## How to Run the Project

### Install required libraries

```bash
pip install -r requirements.txt
```
### Run the project

```bash
python sales_prediction.py
```

## Conclusion

The Linear Regression model successfully predicts product sales with an accuracy of **90.59%**. The analysis shows that **TV and Radio advertising significantly influence sales**, while **Newspaper advertising has comparatively less impact on sales prediction**.

## Author

**Sadhna Kumari**

**CODSOFT Data Science Internship Project**