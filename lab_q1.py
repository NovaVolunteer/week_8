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
avg_price = df.groupby('Neighbourhood ')['Price'].mean()
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
plt.legend()
plt.show()

# %% 
# plot of the log prices by neighborhood 
# log transformation 
df['log_price'] = np.log(df['Price'] + 1) 
# make the KDE plot, grouping the log price column by neighborhood
df.groupby('Neighbourhood ')['log_price'].plot.kde()
plt.title('KDE plot of the log prices by neighborhood')
plt.legend()
plt.show()

# %% [markdown]
# The most expensive borough on average is Manhattan. 

# %% [markdown]
# 2. Regress price on `Neighborhood ` by creating the appropriate dummy/one-hot-encoded variables, without an intercept 
# in the linear model. Compare the coefficients in the regression to the table from part 1. What pattern do you see? 
# What are the coefficients in a regression of a continuous variable on one categorical variable?

# %%
# One-hot encode all of the categorical variables 
# use pd.get_dummies on Neighborhood
df = pd.get_dummies(df,columns=['Neighbourhood '], prefix=['Nbhd'])

# %% 
# make and fit a linear regression model to the one hot encoded neighborhood columns 
# look at the info for the df now to get the new column names 
df.info()

# %%
# set X_simple as the one hot encoded neighborhood columns
X_simple = df[['Nbhd_Bronx', 'Nbhd_Brooklyn', 'Nbhd_Manhattan', 'Nbhd_Queens', 'Nbhd_Staten Island']]
# set y_target as the price column of the data frame
y_target = df['Price']

# %% 
# create the linear regression without an intercept
model = LinearRegression(fit_intercept=False).fit(X_simple, y_target)

# %% 
# print the coefficients of the regression model
print(f'Coefficients: {model.coef_}')

# %% [markdown]
# I can see that the coefficients when I do the linear regression model without intercepts are the exact same for
# each borough as the average price for that borough that I calculated in question one. This means that the 
# coefficients in a regression of a continous variable on one categorical variable is simply the average of that 
# continuous variable for each category. 

# %% [markdown]
# 3. Repeat part 2, but leave an intercept in the linear model. How do you have to handle the creation of the dummies
#  differently? What is the intercept? Interpret the coefficients. How can I get the coefficients in part 2 from these 
# new coefficients?

# %% 
# create X_simple, this time dropping the first one hot encoded column because including all columns would be redundant 
# information with the intercept
# the target is the same, so we don't need to change y_target
X_simple2 = df[['Nbhd_Brooklyn', 'Nbhd_Manhattan', 'Nbhd_Queens', 'Nbhd_Staten Island']]

# %% 
# make a new linear regression model, including the intercept 
model_with = LinearRegression().fit(X_simple2, y_target)

# %% 
# print the coefficients and the intercept 
print(f'Intercept = {model_with.intercept_:.4f}, Coefficents = {model_with.coef_}')

# %% [markdown]
# You handle the creation of the dummies differently by removing the first dummy column from the model because the intercept 
# will give the same information as that column would. The intercept of the model is about 75, which is the same as the 
# average price in the Bronx, the dummy column that was dropped for this model. The coefficents are now not directly the 
# average price in the corresponding neighborhood, but instead are the difference between the average price and the intercept. 
# This means that to get the coefficients in part 2 from these coefifcients, the first coefficient in part 2 is the intercept
# and for the others you simply need to add the intercept to the coefficient in part 3. 

# %% [markdown]
# 4. Split the sample 80/20 into a training and a test set. Run a regression of `Price` on `Review Scores Rating` and 
# `Neighborhood `. What is the $R^2$ and RMSE on the test set? What is the coefficient on `Review Scores Rating`? What
#  is the most expensive kind of property you can rent?

# %% 
# since this regression model is going to include an intercept, we only want to include 4 of the 5 dummy neighborhood 
# columns. We also want to include review scores rating as per the question
X = df[['Review Scores Rating', 'Nbhd_Brooklyn', 'Nbhd_Manhattan', 'Nbhd_Queens', 'Nbhd_Staten Island']]
X_train, X_test, y_train, y_test = train_test_split(X, y_target, train_size=0.8, random_state=42)

# %%
# fit the model on the training set
model4 = LinearRegression().fit(X_train, y_train)

# %% 
# print the R squared value and the RMSE for the test set
y_pred = model4.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'For the test set, R squared = {r2:.4f}, RMSE = {rmse:.3f}')

# %% 
# print the coefficient on Review Scores Rating (this was the first column passed to X, so it will be the first element
#  in the list of coefficients (index 0))
print(f'Coefficient for Review Scores Rating = {model4.coef_[0]:.4f}')

# %% [markdown]
# The R squared for the test set is 0.0459, meaning that not much of the variability in price (for the test set) is 
# represented in this model. The RMSE is 140.918, which means that, on average, the model predicts the price for the 
# test set about $141 away from the actual price. The most expensive type of property you can rent is in Manhattan. 

# %% [markdown]
# 5. Run a regression of `Price` on `Review Scores Rating` and `Neighborhood ` and `Property Type`. What is the $R^2$ 
# and RMSE on the test set? What is the coefficient on `Review Scores Rating`? What is the most expensive kind of 
# property you can rent?

# %% 
# one hot encode Property type since it's a categorical variable. 
df = pd.get_dummies(df, columns=['Property Type'], drop_first=True, prefix=['PT'])

# %%
df.info()

# %%
# since this regression is going to include most of the columns in df, we can make X by dropping the two we don't want.
# y_target is still the same because we're still regressing on price
X2 = df.drop(columns=['Price', 'Room Type', 'Nbhd_Bronx'])

# %% 
# split X and y_target into train and test sets 
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y_target, train_size=0.8, random_state=42)

# %% 
# fit the linear regression model to the new X including all 3 features
model5 = LinearRegression().fit(X2_train, y2_train)

# %% 
# print the R squared, RSME, coefficient on review scores rating, and a list of all coefficients to find the most 
# expensive property type. 
y2_pred = model5.predict(X2_test)
mse = mean_squared_error(y2_test, y2_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y2_test, y2_pred)
print(f'R squared = {r2:.4f}')
print(f'RMSE = {rmse:.4f}')
print(f'Coefficient on Review Scores Rating = {model5.coef_[0]:.4f}')
print(f'All coefficients: {model5.coef_}')

# %% 
X2.info()

# %% [markdown]
# The R squared on the test set is 0.0542, the RMSE on the test set is 140.3027, and the coefficient on Review Scores 
# Rating is 1.2010. The most expensive kind of property you can rent is a bungalow. I know this because PT_Bungalow 
# corresponds to the highest coefficient out of all of the property types, meaning that on average it's the most expensive. 

# %% [markdown]
# 6. What does the coefficient on `Review Scores Rating` mean if it changes from part 4 to 5? Hint: Think about how 
# multiple linear regression works.
# 
# If the coefficient on 'Review Scores Rating" changes from parts 4 to 5, it means that the intercept must have changed.  
# Each coefficient in multiple linear regression represents the change in y when that variable increases by 1, holding 
# all other variables constant. Since all other variables are held constant, 'Review Scores Rating' will only have a 
# different impact if the intercept is different, because this is the only change that's impactful when only looking at 
# a single variable. 
# 
# 7. (Optional) We've included `Neighborhood ` and `Property Type` separately in the model. How do you interact them, 
# so you can have "A bedroom in Queens" or "A townhouse in Manhattan". Split the sample 80/20 into a training and a 
# test set and run a regression including that kind of "property type X neighborhood" dummy, plus `Review Scores 
# Rating`. How does the slope coefficient for `Review Scores Rating`, the $R^2$, and the RMSE change? Do they increase 
# significantly compares to part 5? Are the coefficients in this regression just the sum of the coefficients for `Neighbourhood ` 
# and `Property Type` from 5? What is the most expensive kind of property you can rent?
# %%
