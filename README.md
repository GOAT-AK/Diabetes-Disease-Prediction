# Diabetes-Disease-Prediction
This project focuses on predicting whether a patient has diabetes based on diagnostic measurements. The model is built using Logistic Regression and can be extended to other machine learning classifiers for improved performance.

# Table of Content

* [Brief](#Brief)  
* [DataSet](#DataSet)  
* [How_It_Works](#How_It_Works)  
* [Tools](#Tools)  
* [Model_Performance](#Model_Performance)  
* [Remarks](#Remarks)  
* [Usage](#Usage)

# Brief

Diabetes is a chronic disease that affects millions worldwide. Early detection is crucial for effective management and prevention of complications.  
This project builds a machine learning classification model that uses patient diagnostic measurements to predict whether a patient has diabetes (1) or not (0).

# DataSet

The dataset used in this project is [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data) from Kaggle. The objective is to predict based on diagnostic measurements whether a patient has diabetes.  

All patients are **females of at least 21 years of Pima Indian heritage**.  

### Column Descriptions

| Attribute                        | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| Pregnancies                       | Number of times pregnant                                                    |
| Glucose                           | Plasma glucose concentration a 2 hours in an oral glucose tolerance test    |
| BloodPressure                     | Diastolic blood pressure (mm Hg)                                           |
| SkinThickness                     | Triceps skin fold thickness (mm)                                           |
| Insulin                           | 2-Hour serum insulin (mu U/ml)                                             |
| BMI                               | Body mass index (weight in kg/(height in m)^2)                              |
| DiabetesPedigreeFunction          | Diabetes pedigree function                                                 |
| Age                               | Age (years)                                                                |
| Outcome                           | Class variable (0 = No diabetes, 1 = Diabetes)                             |

# How_It_Works

- Load and clean the dataset.  
- Perform **Exploratory Data Analysis (EDA)** to understand feature distributions and correlations.  
- Apply **feature scaling** using ***StandardScaler*** to normalize numerical features.  
- Split the dataset into **training and testing sets**.  
- Train a **Logistic Regression model**
- Evaluate the model using **accuracy, precision, recall, and F1-score**.  
 

# Tools

I. Jupyter Notebook & VS Code  
II. Python 3.x  
III. pandas, numpy  
IV. matplotlib, seaborn  
V. scikit-learn 


# Model_Performance

The Logistic Regression model was evaluated on the test set.  
Evaluation results showed:  

- **Accuracy:** 82.81%  
- **Classification Report:**  

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 (No Diabetes) | 0.83 | 0.96 | 0.89 | 92 |
| 1 (Diabetes)   | 0.82 | 0.50 | 0.62 | 36 |
| **Overall Accuracy** | **0.83** | | | 128 |
| **Macro Avg**       | 0.82 | 0.73 | 0.75 | 128 |
| **Weighted Avg**    | 0.83 | 0.83 | 0.81 | 128 |

This indicates that the model predicts non-diabetic patients very well but has lower recall for diabetic patients. It can still be useful as a preliminary screening tool, and performance could be improved using feature engineering, hyperparameter tuning, or more advanced classifiers.


# Remarks
* This Python program was run and tested in Jupyter Notebook.  
* Ensure the required libraries are installed by running:

  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn

  # Usage

To begin utilizing this application, follow these steps:

1. Clone this repository:
   
   ```bash
   git clone https://github.com/GOAT-AK/Diabetes-Disease-Prediction

2. Navigate to the cloned repository:

   ```bash
   cd Diabetes-Disease-Prediction

3. Run the Jupyter Notebook:

   ```bash
   Diabetes Disease.ipynb

   
