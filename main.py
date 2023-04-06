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




