# %% [markdown]
# **Q2.** This question is a case study for linear models. The data are about car prices. 
# In particular, they include:
# 
#   - `Price`, `Color`, `Seating_Capacity`
#   - `Body_Type`: crossover, hatchback, muv, sedan, suv
#   - `Make`, `Make_Year`: The brand of car and year produced
#   - `Mileage_Run`: The number of miles on the odometer
#   - `Fuel_Type`: Diesel or gasoline/petrol
#   - `Transmission`, `Transmission_Type`:  speeds and automatic/manual
# 
#   1. Load `cars_hw.csv`. These data were really dirty, and I've already cleaned them 
# a significant amount in terms of missing values and other issues, but some issues 
# remain (e.g. outliers, badly skewed variables that require a log or arcsinh transformation)
#  Note this is different than normalizing: there is a text below that explains further. 
# Clean the data however you think is most appropriate.

# %% 
# load in the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


# %% 
df = pd.read_csv('cars_hw.csv')

# %% 
df.info()

# %% [markdown]
# After looking at the data frame with data wrangler, the unnamed: 0 column can be dropped, as 
# it's the index plus one, the make, color, body_type,fuel_type, transmission, and 
# transmission_type can be turned into categorical variables. These are all strings that have 
# a set number of values, meaning that they're good candidates to become categorical variables 
# and be one-hot-encoded. The No_of_Owners column can also be switched to a numeric column 
# because there's an inherent order in the number of owners that a car has had, meaning that the 
# difference between 1 and 3 versus 1 and 2 is meaningful. Price has some extreme outliers, so we 
# can do a log transformation on it to reduce the infulence of those outliers. I'm choosing to do 
# a log transformation because price is always positive. Since Make_year only 
# ranges from 2011 to 2022, we can center it to be years since 2011.

# %% 
# drop the unnamed: 0 column using df.drop()
df = df.drop('Unnamed: 0', axis=1)

# %% 
# Turn the No_of_Owners to a numeric column, using df.replace and a dictionary of the new values 
df['No_of_Owners'] = df['No_of_Owners'].replace({'1st':1, '2nd':2, '3rd':3})
# change the as type to int 
df['No_of_Owners'] = df['No_of_Owners'].astype(int)

# %% 
# do log transformations on the price column using np.log()
df['Price'] = np.log(df['Price'])

# %% 
# center the Make_year column so it represents years since 2011
df['Make_Year'] = df['Make_Year'] - 2011

# %% 
# turn the string columns to category data type 
cat_cols = ['Make', 'Color', 'Body_Type', 'Fuel_Type', 'Transmission', 'Transmission_Type']
df[cat_cols] = df[cat_cols].astype('category')

# %% [markdown]
#   2. Summarize the `Price` variable and create a kernel density plot. Use `.groupby()`
#  and `.describe()` to summarize prices by brand (`Make`). Make a grouped kernel density
#  plot by `Make`. Which car brands are the most expensive? What do prices look like in 
# general?

# %% 
# group the dataframe by make and describe the price column
df.groupby('Make')['Price'].describe()

# %% 
# make the grouped KDE plot of price based on make 
df.groupby('Make')['Price'].plot.kde()
plt.title('KDE plot of price based on make')
plt.legend()
plt.show()

# %% [markdown]


# %% [markdown]
#   3. Split the data into an 80% training set and a 20% testing set.
#   4. Make a model where you regress price on the numeric variables alone; what is the 
# $R^2$ and `RMSE` on the training set and test set? Make a second model where, for the 
# categorical variables, you regress price on a model comprised of one-hot encoded 
# regressors/features alone (you can use `pd.get_dummies()`; be careful of the dummy 
# variable trap); what is the $R^2$ and `RMSE` on the test set? Which model performs 
# better on the test set? Make a third model that combines all the regressors from 
# the previous two; what is the $R^2$ and `RMSE` on the test set? Does the joint model
#  perform better or worse, and by home much?
#   5. Use the `PolynomialFeatures` function from `sklearn` to expand the set of numerical
#  variables you're using in the regression. As you increase the degree of the expansion, 
# how do the $R^2$ and `RMSE` change? At what point does $R^2$ go negative on the test set? 
# For your best model with expanded features, what is the $R^2$ and `RMSE`? How does it 
# compare to your best model from part 4?
#   6. For your best model so far, determine the predicted values for the test data and 
# plot them against the true values. Do the predicted values and true values roughly line
#  up along the diagonal, or not? Compute the residuals/errors for the test data and 
# create a kernel density plot. Do the residuals look roughly bell-shaped around zero? 
# Evaluate the strengths and weaknesses of your model.