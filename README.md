# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the dataset (Salary.csv).
2. Select the independent variable (Level) as feature X and the dependent variable (Salary) as target y.
3. Split the dataset into training data (70%) and testing data (30%).
4. Predict salaries using the test data and evaluate the model using MAE, MSE, RMSE, and R² score.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: S. KEERTHANA
RegisterNumber:  25004216
*/
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/acer/Downloads/Salary (1).csv")

print("Dataset Preview:")
print(df.head())


X = df[["Level"]]  
y = df["Salary"]             

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MAE  :", mean_absolute_error(y_test, y_pred))
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2_score(y_test, y_pred))
plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=["Level"],
    filled=True
)
plt.title("Decision Tree Regressor for Employee Salary Prediction")
plt.show()
new_exp = [[5]] 
predicted_salary = model.predict(new_exp)
print("\nPredicted Salary for 5 years experience:", predicted_salary[0])
```

## Output:
<img width="821" height="737" alt="image" src="https://github.com/user-attachments/assets/22494891-bccb-4c38-befe-e8853ffe1c0b" />
<img width="821" height="737" alt="image" src="https://github.com/user-attachments/assets/129249c0-6d99-46e0-b584-c586aa9647ae" />




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
