# %% [markdown]
# **Q3.**
# 1. Find a dataset on a topic you're interested in. Some easy options are data.gov, kaggle.com, and data.world.

# %% 
# load in the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import seaborn as sns
import kagglehub

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# %% 
# Download latest version
path = kagglehub.dataset_download("andrewsundberg/college-basketball-dataset")

print("Path to dataset files:", path)
print(os.listdir(path))

# %% 
df = pd.read_csv(os.path.join(path, "cbb.csv"))
df.head()

# %% [markdown]
# 2. Clean the data and do some exploratory data analysis on key variables that 
# interest you. Pick a particular target/outcome variable and features/predictors.
# 
# This dataset is generally pretty clean, with the only missing values being in the 
# postseason and seed columns because not all teams made March Madness, which is where
# these columns were coming from. I want my target variable to be the WAB column, which
# represents the number of wins that a given team has relative to how the average bubble 
# team (the teams that just barely made March Madness) would have performed with their 
# schedule. For my features that I'm planning on including in the model, I want to use 
# statistics from during the games, like adjusted offensive efficiency (ADJOE), adjusted defensive 
# efficiency (ADJDE), effective field goal percentage shot (EFG_O), effective field goal percentage 
# (EFG_D) allowed, Turnover Rate (TOR), Steal Rate (TORD), Offensive Rebound Rate (ORB), 
# Offensive Rebound Rate Allowed (DRB), Free throw rage (FTR), free throw rate allowed (FTRD), 
# and Adjusted Tempo (ADJ_T). 
# 
# This means that I can drop all other columns, as they won't be used in my model. All of the 
# variables I'm keeping look approximately normally distributed when looking at the distribution 
# of values in DataWrangler, meaning that no transformations need to be done on the columns. 
# They also don't have any missing values and are all numeric, meaning that no other cleaning needs 
# to be done. 


# %% 
cbb = df[['ADJOE', 'ADJDE', 'EFG_O', 'EFG_D', 'TOR', 'TORD', 'ORB', 'DRB', 'ADJ_T', 'FTR', 'FTRD', 'WAB']]

# %% 
# look at the description of cbb to see information about these features 
cbb.describe()

# %% 
# I want to make a correlation matrix on the features I'm using and the target variable to ensure 
# that none of them are too correlated and that they will be helpful for my model. 
corr_matrix = cbb.corr('pearson')
corr_with_target = corr_matrix['WAB'].abs().sort_values(ascending=False)

# %% 
sns.heatmap(corr_matrix, 
            annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# From the correlation matrix we can see that none of the variables have a correlation of above 
# 0.9 with the target variable, meaning that we can use all of the features. 

# %% 
# since ADJOE is the most correlated feature with WAB, I want to plot them against eachother to see 
# what the distibution looks like 
# out of curiosity I want to also have the hue represent the posteason column showing where each team 
# finished in march madness (make all null values be NA so matplotlib can assign them a color)
df['POSTSEASON'] = df['POSTSEASON'].fillna('NA')
sns.scatterplot(df, x='ADJOE', y='WAB', hue='POSTSEASON')
plt.xlabel('Adjusted Offensive Efficiency')
plt.ylabel('Wins Above Bubble')
plt.title('Adjusted Offensive Efficiency vs. Wins Above Bubble')
plt.show()
# we can see that the most successful teams in March Madness were also some of the best teams in adjusted 
# offensive efficiency and wins above bubble 

# %% 
# I want to look at the number of turnovers versus the number of steals with hue representing tempo to see 
# if a faster tempo usually leads to more turnovers 
sns.scatterplot(cbb, x='TOR', y='TORD', hue='ADJ_T')
plt.xlabel('Turnover Rate')
plt.ylabel('Steal Rate')
plt.title('Turnover rate vs. steal rate by tempo')
plt.show()
# this is not helpful at all 

# %% 
# adjusted defensive efficiency is strongly negatively correlated with WAB so I want to look at this relationship
# I also want to make the hue the adjusted offensive efficiency to see how these three work together 
sns.scatterplot(cbb, x='ADJDE', y='WAB', hue='ADJOE')
plt.xlabel('Adjusted Defensive Efficiency')
plt.ylabel('Wins Above Bubble')
plt.title('Adjusted defnesive efficiency vs. wins above bubble')
plt.show()
# you can see a clear correlation with lower ADJDE (average number of points allowed against the average D1 offense)
# and higher WAB, along with teams that have a higher WAB havign a higher ADJOE even if their ADJDE isn't very low. 

# %% 
# I want to plot ADJOE and ADJDE to see if they're correlated, since the plot above makes it seem like they're both 
# correlated with WAB more 
sns.scatterplot(cbb, x='ADJOE', y='ADJDE')
plt.xlabel('Adjusted offensive efficiency')
plt.ylabel('Adjusted defensive efficiency')
plt.title('Adjusted offensive efficiency vs. adjusted defensive efficiency')
plt.show()
# you see a general trend that teams with a higher ADJOE have a lower ADJDE, but it's less correlated
# than the graphs above 

# %% [markdown]
# 3. Split the sample into an ~80% training set and a ~20% test set.

# %% 
# use train_test_split to make our train and test sets with a train size of 0.8 
# set the random state so we get the same split every time 
train, test = train_test_split(cbb, train_size=0.8, random_state=42)

# %% [markdown]
# 4. Run a few regressions of your target/outcome variable on a variety of 
# features/predictors. Compute the RMSE on the test set.

# %% 
# first, I want to make a model only with ADJOE and ADJDE since they are the most correlated features to WAB 
# according to the correlation matrix 
X = train[['ADJOE', 'ADJDE']]
y = train['WAB']
# also make the test data frames to get the $R^2$ and RMSE on the test set 
X_test = test[['ADJOE', 'ADJDE']]
y_test = test['WAB']

# %% 
# I'm going to use a for loop with polynomial features to see if higher degree functions would be better 
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
# We can see that when just looking at ADJOE and ADJDE compared to WAB, a degree 3 polynomial works best. 
# So far, our $R^2$ and RMSE to beat are 0.8891 and 2.317 respectively. 

# %% 
# now I want to take only the scoring features included in cbb (EFG_O, EFG_D, FTR, FTRD) into account 
# in the model to see how that compares 
X = train[['EFG_O', 'EFG_D', 'FTR', 'FTRD']]
y = train['WAB']
# make the test set 
X_test = test[['EFG_O', 'EFG_D', 'FTR', 'FTRD']]
y_test = test['WAB']

# %%
# use a for loop to test regression with up to 10 polynomial features  
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
# We can see that this model performs worse than the first one, especially at higher degrees

# %% 
# run the same loop, making the model use all features in cbb
X = train.drop('WAB', axis=1) 
y = train['WAB']
# make same X and y using the test set 
X_test = test.drop('WAB', axis=1)
y_test = test['WAB']

# %% 
# use a lower range since this has more features so that the kernel doesn't crash 
for i in range(4): 
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
# This gives us our best-performing model so far (degree 2), with a $R^2$ on the test set of 0.9113
# and a RMSE of 2.071. This RMSE is very good, as the range of wins against bubble ranges from 
# -25.2 to 13.1, a range of 38.3, meaning that the RMSE of 2.071 will give a very good idea of
# the number of wins against bubble a team will have. To see the effectiveness of this model 
# against more data, I also want to test it on the data from 2026 which includes all of the games 
# up until March Madness started. 

# %% 
cbb26 = pd.read_csv(os.path.join(path, "cbb26.csv"))
cbb26 = cbb26[['ADJOE', 'ADJDE', 'EFG_O', 'EFG_D', 'TOR', 'TORD', 'ORB', 'DRB', 'ADJ_T', 'FTR', 'FTRD', 'WAB']]

# %%
X_test = cbb26.drop('WAB', axis=1)
y_test = cbb26['WAB']

# %% 
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
pol = LinearRegression().fit(X_poly, y)
X_poly_test = poly.transform(X_test)
y_pred_poly = pol.predict(X_poly_test)
mse = mean_squared_error(y_test, y_pred_poly)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_poly)
print(f'R squared = {r2:.4f}, RMSE = {rmse:.3f}')

# %% [markdown]
# This model actually works even better on the data from 2026! 

# %% [markdown]
# 5. Which model performed the best, and why?
# 6. What did you learn?