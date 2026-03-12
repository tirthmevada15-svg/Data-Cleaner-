# Patient Health Data Cleaning & Preprocessing Project

## Overview

This project demonstrates a complete **data preprocessing pipeline for
healthcare data**.\
The goal is to analyze a patient dataset, handle missing values using
multiple imputation techniques, detect and treat outliers, and prepare a
clean dataset ready for machine learning models.

The project is implemented in **Python using Pandas, Scikit‑Learn,
NumPy, and visualization libraries**.

------------------------------------------------------------------------

# Dataset Description

The dataset contains synthetic patient health records.

  Column           Description
  ---------------- -----------------------------------------------
  patient_id       Unique patient identifier
  age              Age of patient
  gender           Male / Female
  region           Patient location (North, South, East, West)
  bmi              Body Mass Index
  blood_pressure   Systolic blood pressure
  cholesterol      Cholesterol level
  glucose          Blood glucose level
  disease_risk     Target variable (0 = Low risk, 1 = High risk)

The dataset intentionally contains: - Missing values - Synthetic
outliers

This allows demonstration of **data cleaning techniques**.

------------------------------------------------------------------------

# Project Workflow

The project follows a standard **data science preprocessing pipeline**.

1.  Dataset Loading
2.  Exploratory Data Analysis
3.  Missing Value Analysis
4.  Missing Value Imputation
5.  Outlier Detection
6.  Outlier Treatment
7.  Visualization
8.  Export Clean Dataset

------------------------------------------------------------------------

# Missing Value Handling Techniques

The following techniques were used:

  Technique                    Application
  ---------------------------- -------------------------
  Simple Imputer (Median)      BMI values
  Most Frequent Imputer        Gender and Region
  Random Sampling Imputation   Cholesterol
  KNN Imputer                  Age, Glucose, BMI
  MICE (Iterative Imputer)     Multivariate imputation

These techniques ensure missing values are handled without significantly
distorting the dataset.

------------------------------------------------------------------------

# Outlier Detection Methods

Several statistical methods were used to detect abnormal values.

### Z‑Score Method

Detects observations more than **3 standard deviations** from the mean.

### IQR Method

Uses interquartile range:

Lower Bound = Q1 − 1.5 × IQR\
Upper Bound = Q3 + 1.5 × IQR

### Percentile Method

Extreme values capped at **1st and 99th percentile**.

### Winsorization

Extreme values replaced with boundary values to preserve dataset size.

------------------------------------------------------------------------

# Visualizations

The project includes:

-   Missing value heatmap
-   BMI boxplot
-   Cholesterol distribution
-   Correlation heatmap

These help understand data distribution and anomalies.

------------------------------------------------------------------------

# Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit‑Learn
-   Seaborn
-   Matplotlib
-   SciPy

------------------------------------------------------------------------

# Project Structure

    patient-health-data-project
    │
    ├── patient_health_dataset.csv
    ├── patient_health_cleaned.csv
    ├── data_cleaning_notebook.ipynb
    ├── main.py
    └── README.md

------------------------------------------------------------------------

# How to Run the Project

### 1. Install Required Libraries

    pip install pandas numpy matplotlib seaborn scikit-learn scipy

### 2. Run the Script

    python main.py

or run the Jupyter notebook.

------------------------------------------------------------------------

# Output

After running the project, a cleaned dataset is generated:

    patient_health_cleaned.csv

This dataset is ready for:

-   Machine learning
-   Predictive modeling
-   Health risk analysis

------------------------------------------------------------------------

# Future Improvements

Possible extensions:

-   Add machine learning model for disease prediction
-   Build a healthcare analytics dashboard
-   Deploy model as an API
-   Create interactive data visualization
