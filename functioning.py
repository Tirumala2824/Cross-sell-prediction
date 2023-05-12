# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from LogisticRegreesion import Logisticregression
from train_test_split import train_test_split1
import random

# from LogisticRegreesion import LogisticRegression
# from sklearn.model_selection import train_test_split

# Load data into a pandas dataframe
train = pd.read_csv(r'D:\data analysis project\pythonProject 1\Semester_project_6thsem\train.csv')
train_data = pd.DataFrame(train)

test = pd.read_csv(r'D:\data analysis project\pythonProject 1\Semester_project_6thsem\test.csv')
test_data = pd.DataFrame(test)

train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True)

print(data.describe())
print(data.columns)

# print(data.isnull().sum())

# Let us plot a histogram for each numerical variable and analyze the distribution.


data.hist(figsize=(15, 10), layout=(3, 3))
plt.tight_layout()
# plt.show()

# plt.close()

data.drop('Driving_License', axis=1, inplace=True)

# Numerical and Categorical variables analysis

sns.barplot(x='Gender', y='Response', data=data)
plt.title('Gender')
# plt.show()

sns.barplot(x='Vehicle_Age', y='Response', data=data)
# plt.show()

# breakpoint()
gender_map = {'Male': 0, 'Female': 1}
data['Gender'] = data['Gender'].map(gender_map)

vehicle_age_map = {'1-2 Year': 0, '< 1 Year': 1, '> 2 Years': 2}
data['Vehicle_Age'] = data['Vehicle_Age'].map(vehicle_age_map)
Vehicle_Damage_map = {'Yes': 0, 'No': 1}
data['Vehicle_Damage'] = data['Vehicle_Damage'].map(Vehicle_Damage_map)

X = data[['id', 'Gender', 'Age', 'Region_Code','Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium','Policy_Sales_Channel', 'Vintage']]
y = data['Response']

# values={"Response":1.0}
Y=y.fillna(0)
print(Y)
print(type(X),type(Y))
# breakpoint()
train_size = int(0.8 * len(X))
print(train_size)
x_train, y_train = X[:train_size], Y[:train_size]
x_test, y_test = X[train_size:], Y[train_size:]




