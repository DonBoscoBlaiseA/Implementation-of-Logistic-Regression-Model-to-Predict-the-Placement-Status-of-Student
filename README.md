# Ex.4 Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use LabelEncoder to transform categorical features (gender, ssc_b, hsc_b, etc.) into numerical labels.
2. Define features (x) and target variable (y) using data1, excluding the "status" column.
3. Split the dataset into training and testing sets with a specified test size and random state.
4. Initialize and train a LogisticRegression model with "liblinear" solver using the training data.
5. Predict status for the testing data, generate a classification report using sklearn.metrics.classification_report, and make predictions for new input features.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Don Bosco Blaise A
RegisterNumber: 212221040045
*/

import pandas as pd
data=pd.read_csv("G:/jupyter_notebook_files/placement_data/Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![Screenshot (224)](https://github.com/DonBoscoBlaiseA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/140850829/12b82fc8-39e8-439a-90ad-ee60d4f96ed2)
![Screenshot (225)](https://github.com/DonBoscoBlaiseA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/140850829/ffc2befe-6854-40e5-a4d4-067c6c2b3921)
![Screenshot (226)](https://github.com/DonBoscoBlaiseA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/140850829/d8416629-d2b4-4a4b-bbf2-e740518a992d)
![Screenshot (227)](https://github.com/DonBoscoBlaiseA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/140850829/7b447260-5189-4531-ae36-4c0b1a6e6676)
![Screenshot (228)](https://github.com/DonBoscoBlaiseA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/140850829/3613506b-18c1-459d-a472-54d124313223)
![Screenshot (229)](https://github.com/DonBoscoBlaiseA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/140850829/38f4d4ba-d204-4e7d-b663-21efed674c11)
![Screenshot (230)](https://github.com/DonBoscoBlaiseA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/140850829/d720beca-57c0-416c-a5e0-c62bdc4795da)
![Screenshot (231) 1](https://github.com/DonBoscoBlaiseA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/140850829/384004e1-e1d2-49ef-b387-5175a84e0bef)
![Screenshot (231)](https://github.com/DonBoscoBlaiseA/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/140850829/35a81cda-d603-45d5-b916-32252a262564)
<br>
<br>
<br>
<br>
<br>
<br>
<br>  

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
