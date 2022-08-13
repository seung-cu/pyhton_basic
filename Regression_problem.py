# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import os
import warnings
warnings.filterwarnings('ignore')



# %% 
## Function List 


# %% 
## Data Loading

train = pd.read_csv("./Dataset/Regression/train.csv")
test = pd.read_csv("./Dataset/Regression/test.csv")

# %% 
## EDA
print("length of data:", len(train))
print(train.head())
print(train.info())
print(train.isnull().sum())
print(test.isnull().sum())

# %%
train["datetime"] = pd.to_datetime(train["datetime"])
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["weekday"] = train["datetime"].dt.weekday ## 0: Mon ~ 6 : Sun

# %%

train["season"] = train["season"].map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })
train["weekday"] = train["weekday"].map({0: "Mon", 1:"Tue", 2:"Wen", 3:"Thr", 4:"Fri", 5:"Sat", 6:"Sun"})
train["weather"] = train["weather"].map({1: "Clear + Few clouds",\
                                        2 : "Mist + Cloudy ", \
                                        3 : "Light Snow + Rain", \
                                        4 : "Heavy Rain + Ice Pallets" })


# %%
categorical_variable = ["season", "holiday", "workingday", "weather", "year", "month", "day", "weekday"]
numerical_variable = ["temp", "atemp", "humidity", "windspeed", "casual", "registered"]
date_variable = ["datetime"]
target_variable =["count"]

# %%
## Visualization of Categorical variable vs Target variable(count, plot)
for cat in categorical_variable:
    train[cat] =train[cat].astype("category")

# %%
for i, cat in enumerate(categorical_variable):    
    print(cat)
    temp = train.groupby(cat)["count"].mean().reset_index()
    sns.barplot(x= cat, y ='count', data = temp)
    plt.show()
# %%
for i, cat in enumerate(categorical_variable):    
    print(cat)
    temp = train[["count", cat]]
    model = ols('count ~ C({})'.format(cat), temp).fit()
    p_value = anova_lm(model)["PR(>F)"][0]
    if p_value < 0.05:
        print(cat ,": siginifcance -> all population are not equal")
        print("------------")
    else:
        print(cat, ":not-significant -> all population are equal")
        print("------------")

   
# %%
## Visualization of numerical variable vs Target variable(Distribution)

# %%
## Outlier analysis for Target variable


# %%  Sample
dir(anova_lm(model))
# %%
anova_lm(model).items()["p-Value"]
# %%
anova_lm(model)._get_value
# %%
anova_lm(model)["PR(>F)"].iloc[0]
# %%


# %%

# %%
