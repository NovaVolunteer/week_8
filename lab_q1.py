# %% [markdown]
# **Q1.** Load clean q1_clean: https://raw.githubusercontent.com/DS3001/linearRegression/refs/heads/main/data/Q1_clean.csv 
# 
# The data include
# 
# - `Price` per night
# - `Review Scores Rating`: The average rating for the property
# - `Neighborhood `: The bourough of NYC. Note the space, or rename the variable.
# - `Property Type`: The kind of dwelling
# - `Room Type`: The kind of space being rented
# 

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
# read in the data 
url = 'https://raw.githubusercontent.com/DS3001/linearRegression/refs/heads/main/data/Q1_clean.csv' 
df = pd.read_csv(url)

# %% 
# look at the info for the data 
df.info()

# %% 
df.columns

# %% [markdown]
# 1. Compute the average prices and scores by `Neighborhood `; which borough is the most expensive on average? Create a 
# kernel density plot of price and log price, grouping by `Neighborhood `.

# %% 
# group the data by neighborhood and then look at the prices and scores columns to get the averages by neighborhood
avg_price = df.groupby('Neighbourhood ').mean()
avg_score = df.groupby('Neighbourhood ')['Review Scores Rating'].mean()

# %% 
avg_price
# The neighborhood with the highest average price is Manhattan. 

# %% 
avg_score
# The neighborhood with the highest average score is Brooklyn. 

# %% 
# KDE plot of the price grouped by neighborhood
df.groupby('Neighbourhood ')['Price'].plot.kde()
plt.title('KDE plot of the prices based by neighborhood')
plt.show()

# %% 
# plot of the log prices by neighborhood 
# log transformation 
df['log_price'] = np.log(df['Price'] + 1) 
# make the KDE plot, grouping the log price column by neighborhood
df.groupby('Neighbourhood ')['log_price'].plot.kde()
plt.title('KDE plot of the log prices by neighborhood')
plt.show()

# %% [markdown]
# 2. Regress price on `Neighborhood ` by creating the appropriate dummy/one-hot-encoded variables, without an intercept 
# in the linear model. Compare the coefficients in the regression to the table from part 1. What pattern do you see? 
# What are the coefficients in a regression of a continuous variable on one categorical variable?

# %%
cat_cols = ['Neighbourhood ', 'Property Type', 'Property Type']
df = pd.get_dummies(df,columns=cat_cols, drop_first=True)

# %% [markdown]
# 3. Repeat part 2, but leave an intercept in the linear model. How do you have to handle the creation of the dummies
#  differently? What is the intercept? Interpret the coefficients. How can I get the coefficients in part 2 from these 
# new coefficients?
# 4. Split the sample 80/20 into a training and a test set. Run a regression of `Price` on `Review Scores Rating` and 
# `Neighborhood `. What is the $R^2$ and RMSE on the test set? What is the coefficient on `Review Scores Rating`? What
#  is the most expensive kind of property you can rent?
# 5. Run a regression of `Price` on `Review Scores Rating` and `Neighborhood ` and `Property Type`. What is the $R^2$ 
# and RMSE on the test set? What is the coefficient on `Review Scores Rating`? What is the most expensive kind of 
# property you can rent?
# 6. What does the coefficient on `Review Scores Rating` mean if it changes from part 4 to 5? Hint: Think about how 
# multiple linear regression works.
# 7. (Optional) We've included `Neighborhood ` and `Property Type` separately in the model. How do you interact them, 
# so you can have "A bedroom in Queens" or "A townhouse in Manhattan". Split the sample 80/20 into a training and a 
# test set and run a regression including that kind of "property type X neighborhood" dummy, plus `Review Scores 
# Rating`. How does the slope coefficient for `Review Scores Rating`, the $R^2$, and the RMSE change? Do they increase 
# significantly compares to part 5? Are the coefficients in this regression just the sum of the coefficients for `Neighbourhood ` 
# and `Property Type` from 5? What is the most expensive kind of property you can rent?