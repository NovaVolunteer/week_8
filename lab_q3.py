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
# 2. Clean the data and do some exploratory data analysis on key variables that interest you. Pick a particular target/outcome variable and features/predictors.
# 3. Split the sample into an ~80% training set and a ~20% test set.
# 4. Run a few regressions of your target/outcome variable on a variety of features/predictors. Compute the RMSE on the test set.
# 5. Which model performed the best, and why?
# 6. What did you learn?