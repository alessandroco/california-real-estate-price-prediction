# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the data
data = pd.read_csv('housing.csv')

# Dropping rows with missing values and saving the changes.
data.dropna(inplace=True)


# Splitting the data into X & Y data.
    # X will be the same data frame but without the target Variable (Median House Value).
        # The "Axis=1" command will make us drop the whole column.
    # Y logically will be only the target variable (Median House Value).

from sklearn.model_selection import train_test_split
x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

# We'll now set the ratios of both the Learning data and the Testind data with an optional ratio of 0.2 of the data used for testing.

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

