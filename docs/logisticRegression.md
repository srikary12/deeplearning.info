# Logistic Regression

## Purpose

- Helps in classifying a label
- I think of it as a linear classifier which draws a regression line and classifies the point as above or below the line
- Gives you a probibility of belonging to ones class

## General use cases

- Email spam detection: Classify emails as spam or not spam 
- Medical diagnosis: Predict medical conditions based on patient data 
- Fraud detection: Identify data anomalies that indicate fraud 
- Insurance policy approval: Decide whether to approve a new policy based on a driver's history and credit history 


## Implementation
### We are not selecting the skinthickness as it is not required as per domain knowledge. Will write on feature selection in a separate page
```py
import kagglehub

path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
print("Path to dataset files:", path)

# Read csv and update the column names
import pandas as pd

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label'] 
pima = pd.read_csv(f"{path}" + "/diabetes.csv", sep=",", names=col_names) # Replcing names to make it easier to work with

#EDA
import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.plot(df["experience"], df["income"], 'ro')
plt.subplot(2,1,2)
plt.plot(df["age"], df["income"], 'ro')
plt.show()
```
![Plot showing data](img\logistic_reg\eda.png)

### What can we observe from the graph

- Most people are at the low insulin levels.
- Number of pregnencies are <= 2 for more than 50% of people.
- The Major age group of the dataset is between 0 - 35.
- We can observe that the skin thickness can be ignored as most of the people have a similar skin thickness


=== "Python"
    ```py
    ```

=== "sklearn"
    ```py
    ```

=== "Pytorch"
    ```py
    ```
