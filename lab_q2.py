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
df['log_Price'] = np.log(df['Price'])

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
# MG motors is the most expensive overall with Kia and Jeep also being expensive. In general, 
# prices seem to be around 60-70 thousand dollars. 

# %% [markdown]
#   3. Split the data into an 80% training set and a 20% testing set.

# %% 
# one-hot-encode the data frame before we split it so that we can use the categorical variables later
cat = list(df.select_dtypes(include=['category']).columns)
# make a new data frame with these one hot encoded 
df= pd.get_dummies(df, columns=cat, drop_first=True)

# %% 
# use train_test_split to split the data into train and test sets 
train, test = train_test_split(df, train_size=0.8, random_state=42)

# %% [markdown]
#   4. Make a model where you regress price on the numeric variables alone; what is the 
# $R^2$ and `RMSE` on the training set and test set? Make a second model where, for the 
# categorical variables, you regress price on a model comprised of one-hot encoded 
# regressors/features alone (you can use `pd.get_dummies()`; be careful of the dummy 
# variable trap); what is the $R^2$ and `RMSE` on the test set? Which model performs 
# better on the test set? Make a third model that combines all the regressors from 
# the previous two; what is the $R^2$ and `RMSE` on the test set? Does the joint model
#  perform better or worse, and by how much?

# %%
# df.info()
# make X the numeric variables from the train set 
num_X = train[['Make_Year', 'Mileage_Run', 'No_of_Owners', 'Seating_Capacity']]
# make y the converted price variable, our target 
target_y = train['log_Price']

# %% 
# initalize the linear regression model, fitting it to our X and y from the train set
model_num = LinearRegression().fit(num_X, target_y)

# %%
# find the R squared and the RMSE for the train set 
y_pred = model_num.predict(num_X)
mse = mean_squared_error(target_y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(target_y, y_pred)
print(f'For the train set with the numeric model, R squared = {r2:.4f}, RMSE = {rmse:.3f}')

# %% 
# find the R squared and RMSE for the test set
# select the columns for X that are used in the regression from the test set
test_X = test[['Make_Year', 'Mileage_Run', 'No_of_Owners', 'Seating_Capacity']]
# select the target variable from the test set
test_y = test['log_Price']

# %% 
y_pred = model_num.predict(test_X)
mse = mean_squared_error(test_y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(test_y, y_pred)
print(f'For the test set with the numeric model, R squared = {r2:.4f}, RMSE = {rmse:.3f}')

# %% 
# train.info()
# make another model based on the categorical variables 
# let X_cat equal all of the boolean columns which are the columns housing the dummy columns
# make a list cols of the boolean columns
cols = list(train.select_dtypes(include=['bool']).columns)
X_cat = train[cols]
# let y_target equal the log_Price column from the train data frame 
y_target = train['log_Price']

# %% 
# initalize and fit the new model 
model_cat = LinearRegression().fit(X_cat, y_target)

# %%
# find the R squared and the RMSE for the train set 
y_pred = model_cat.predict(X_cat)
mse = mean_squared_error(target_y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(target_y, y_pred)
print(f'For the train set with the categorical model, R squared = {r2:.4f}, RMSE = {rmse:.3f}')

# %%
test_cat = test[cols]
y_test = test['log_Price']

# %% 
# find the R squared and the RMSE for the train set 
y_pred = model_cat.predict(test_cat)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'For the test set with the categorical model, R squared = {r2:.4f}, RMSE = {rmse:.3f}')

# %% 
# combine both models - use all of the columns except for price and log_Price 
# make a list of all columns except for price and log_price 
cols = list(train.drop(['Price', 'log_Price'], axis=1).columns)
X = train[cols]
# make y as the log price column 
y_target = train['log_Price']

# %% 
# initalize and fit the model 
model = LinearRegression().fit(X, y_target)

# %% 
# find the R squared and the RMSE for the train set 
y_pred = model.predict(X)
mse = mean_squared_error(y_target, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_target, y_pred)
print(f'For the train set with the joint model, R squared = {r2:.4f}, RMSE = {rmse:.3f}')

# %% 
# make X and y based on the test set
X_test = test[cols]
y_test = test['log_Price']

# %%
# find the R squared and the RMSE for the train set 
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'For the test set with the joint model, R squared = {r2:.4f}, RMSE = {rmse:.3f}')

# %% [markdown]
# The joint model performs better than the other two by a pretty signifigant amount. The R 
# squared for the test set is 0.81, which is larger than the R squared of 0.65 and 0.38 for 
# the categorical and numeric models respectively. The root mean squared error is also lower, 
# being 0.188 (log value) compared to 0.257 and 0.345. This means that the joint model is 
# pretty clearly the best model to use. 

# %% [markdown]
#   5. Use the `PolynomialFeatures` function from `sklearn` to expand the set of numerical
#  variables you're using in the regression. As you increase the degree of the expansion, 
# how do the $R^2$ and `RMSE` change? At what point does $R^2$ go negative on the test set? 
# For your best model with expanded features, what is the $R^2$ and `RMSE`? How does it 
# compare to your best model from part 4?


# %% 
X = train[['Make_Year', 'Mileage_Run', 'No_of_Owners', 'Seating_Capacity']]
# make y as the log price column 
y = train['log_Price']
X_test = test[['Make_Year', 'Mileage_Run', 'No_of_Owners', 'Seating_Capacity']]
y_test = test['log_Price']
# use a for loop to print the R squared and the RMSE for all degrees 1-10
for i in range(10): 
    poly = PolynomialFeatures(degree=i+1, include_bias=False)
    X_poly = poly.fit_transform(X)
    pol = LinearRegression().fit(X_poly, y)
    X_poly_test = poly.transform(X_test)
    y_pred_poly = pol.predict(X_poly_test)
    mse = mean_squared_error(y_test, y_pred_poly)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_poly)
    print(f'For degree {i+1}, R squared = {r2:.4f}, RMSE = {rmse:.3f}')

# %% [markdown]
# When I increase the degree, the $R^2$ increases from degree 1 to 2 and from degree 2 to 3
# but decreases from degree 3 to 4 and keeps decreasing after that. The RMSE follows the same 
# pattern but inverted, with the RMSE decreasing from degree 1 to 2 and 2 to 3, but increases 
# after that. This means that the model gets better at predicting based on the numeric variables 
# when it has degree 2 or 3, but it gets less accurate when the degree is larger than 3. The 
# $R^2$ goes negative on the test set when the model is degree 10. The best model with these 
# expanded features is of degree 3, which has a $R^2$ of 0.4179 and a RMSE of 0.333. These are 
# both worse than my best model from question 4, as that has a $R^2$ of 0.8139 and a RMSE of 0.188.


# %% [markdown]
#   6. For your best model so far, determine the predicted values for the test data and 
# plot them against the true values. Do the predicted values and true values roughly line
#  up along the diagonal, or not? Compute the residuals/errors for the test data and 
# create a kernel density plot. Do the residuals look roughly bell-shaped around zero? 
# Evaluate the strengths and weaknesses of your model.

# %% 
# take the best model and use it to make predictions on the test set 

# make the X and y of our best model 
# combine both models - use all of the columns except for price and log_Price 
# make a list of all columns except for price and log_price 
cols = list(train.drop(['Price', 'log_Price'], axis=1).columns)
X = train[cols]
# make y as the log price column 
y_target = train['log_Price']

# %% 
# initalize and fit the model 
model = LinearRegression().fit(X, y_target)

# %% 
# make X and y based on the test set
X_test = test[cols]
y_test = test['log_Price']

# %%
# find the predicted values for y based on the test set 
y_pred = model.predict(X_test)

# %% 
# plot the true values and the predicted values in a scatter plot 
plt.scatter(y_test, y_pred)
plt.title('True values vs. predicted values')
plt.xlabel('True values (log price)')
plt.ylabel('Predicted values (log price)')
# add a diagonal line that runs from the lowest value in y_test to the highest (gives the 
# line of perfect predictions since the X and Y values are equal)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')
plt.show()

# %% [markdown]
# The scatter plot of the true and predicted values does cluster around the diagonal line. 

# %% 
# residuals plot 
resid  = y_test - y_pred
sns.kdeplot(resid, fill=True)
plt.axvline(0, color='red', linestyle='--') 
plt.title("KDE Plot of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Density")
plt.show()

# %% [markdown]
# The KDE plot of residuals does look roughly bell shaped around 0. It's a little bit skewed, 
# but the overall shape is quite close to a bell curve and it's still centered close to 0. 
# A strength of my model is that it's decently good at predicting the log price of a car based 
# on the information about the car. It covers about 81% of the variability in the data and the 
# root mean squared error is 0.188 while the log prices range from about 12.5 to 14.5. This means 
# that the prediction the model gives is usually within a 10% range of the log prices, which is 
# better than if a human who didn't know about cars tried to make a prediction. A weakness of the 
# model is that it includes a lot of features, as most of the features in the original data set 
# were categorical, meaning that dummy variables had to be made for them and some of them, like make,
# had a lot of possible values, meaning that a lot of dummy variables were created for them. This 
# means that there's a chance it was overfit to the data, however, the model still performed the best
# out of all the ones tested on the test data. 