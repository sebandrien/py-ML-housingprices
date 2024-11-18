import pandas as pd                              
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("housing.csv") #Reading in housing file

data #Visualzing data

data.info() #Displaying "data" info

data.dropna(inplace=True) #Dropping null values from "data", done to properly train model

data.info() #Displaying new "data"

from sklearn.model_selection import train_test_split
x = data.drop(['median_house_value'],axis=1)
y = data['median_house_value']

x_train,x_test,y_train,y_text = train_test_split(x,y,test_size=0.2)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('d.csv')
print(data.columns)

plt.figure(figsize=(10, 6))
sns.histplot(data['ocean_proximity'], bins=20, kde=True)
plt.title("Ocean Proximity")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['median_house_value'], bins=20, kde=True)
plt.title("Median House Value")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['median_house_value'], bins=20, kde=True)
plt.title("Median House Value")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['median_house_value'], bins=20, kde=True)
plt.title("Median House Value")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['median_income'], bins=20, kde=True)
plt.title("Median House Value")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='median_income', y='median_house_value', data=data, hue='ocean_proximity', palette='viridis')
plt.title("Median Income vs Median House Value (Colored by Ocean Proximity)")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='population', y='median_house_value', data=data, hue='ocean_proximity', palette='viridis', s=100)
plt.title("Median House Value by Ocean Proximity", fontsize=16)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='population', y='median_house_value', data=data, hue='ocean_proximity', palette='viridis', s=100)
plt.title("Median House Value by Ocean Proximity", fontsize=16)
plt.show()

numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()




