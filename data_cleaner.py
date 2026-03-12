# 1. Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from scipy import stats
from scipy.stats.mstats import winsorize

# 2. Load Dataset
df = pd.read_csv("patient_health_dataset.csv")

print("Dataset Loaded Successfully\n")

print("First 5 Rows\n")
print(df.head())

# 3. Dataset Information
print("\nDataset Shape:", df.shape)

print("\nDataset Info")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# 4. Missing Value Analysis
missing_count = df.isnull().sum()

missing_percent = (df.isnull().sum()/len(df))*100

missing_report = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percent
})

print("\nMissing Value Report\n")
print(missing_report)

# 5. Missing Value Visualization
plt.figure(figsize=(10,6))

sns.heatmap(df.isnull(), cbar=False)

plt.title("Missing Values Heatmap")

plt.show()

# 6. Handling Missing Values
# 6.1 Simple Imputer for BMI
median_imputer = SimpleImputer(strategy="median")

df["bmi"] = median_imputer.fit_transform(df[["bmi"]])

# 6.2 Most Frequent Imputer for Categorical
cat_imputer = SimpleImputer(strategy="most_frequent")

df[["gender","region"]] = cat_imputer.fit_transform(df[["gender","region"]])

# 6.3 Missing Indicator + Random Sampling
df["cholesterol_missing"] = df["cholesterol"].isnull().astype(int)

random_values = df["cholesterol"].dropna().sample(df["cholesterol"].isnull().sum(), replace=True)

df.loc[df["cholesterol"].isnull(),"cholesterol"] = random_values.values

# 6.4 KNN Imputer
num_cols = ["age","bmi","blood_pressure","cholesterol","glucose"]

knn_imputer = KNNImputer(n_neighbors=5)

df[num_cols] = knn_imputer.fit_transform(df[num_cols])

# 6.5 MICE Imputation
mice_imputer = IterativeImputer(random_state=42)

df[num_cols] = mice_imputer.fit_transform(df[num_cols])


print("\nMissing Values After Imputation\n")

print(df.isnull().sum())

# 7. Outlier Detection
# 7.1 Z Score Method
z_scores = np.abs(stats.zscore(df[num_cols]))

outliers = np.where(z_scores > 3)

print("\nTotal Outliers Detected using Z-score:", len(outliers[0]))

# 8. IQR Method (BMI)
Q1 = df["bmi"].quantile(0.25)

Q3 = df["bmi"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR

bmi_outliers = df[(df["bmi"] < lower) | (df["bmi"] > upper)]

print("\nBMI Outliers using IQR:", len(bmi_outliers))

# 9. Percentile Method
lower = df["cholesterol"].quantile(0.01)

upper = df["cholesterol"].quantile(0.99)

df["cholesterol"] = np.clip(df["cholesterol"], lower, upper)

# 10. Winsorization
df["glucose"] = winsorize(df["glucose"], limits=[0.01,0.01])

# 11. Visualization
plt.figure(figsize=(6,4))

sns.boxplot(x=df["bmi"])

plt.title("BMI Distribution After Outlier Treatment")

plt.show()


plt.figure(figsize=(6,4))

sns.histplot(df["cholesterol"], bins=30)

plt.title("Cholesterol Distribution")

plt.show()

# 12. Correlation Matrix
plt.figure(figsize=(8,6))

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")

plt.title("Feature Correlation")

plt.show()

# 13. Final Dataset Summary
print("\nFinal Dataset Summary\n")

print(df.describe())

# 14. Save Clean Dataset
df.to_csv("patient_health_cleaned.csv", index=False)

print("\nClean dataset saved successfully as patient_health_cleaned.csv")

# End of Project