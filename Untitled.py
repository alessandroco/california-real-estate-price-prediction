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

train_data ["total_rooms"] = np.log(train_data['total_rooms'] + 1 )
train_data ["total_bedrooms"] = np.log(train_data['total_bedrooms'] + 1 )
train_data ["population"] = np.log(train_data['population'] + 1 )
train_data ["households"] = np.log(train_data['households'] + 1 )


# Now we want to join the dummies to our dataset 

train_data.join(pd.get_dummies(train_data.ocean_proximity, dtype=int))

# and drop the ocean proximity column entirely

train_data.join(pd.get_dummies(train_data.ocean_proximity, dtype=int)).drop(['ocean_proximity'], axis=1)