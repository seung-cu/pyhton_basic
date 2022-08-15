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
categorical_variable = ["season", "holiday", "workingday", "weather", "year", "month", "day", "weekday" ,"hour"]
numerical_variable = ["temp", "atemp", "humidity", "windspeed"]
date_variable = ["datetime"]
semi_target_variable = ["casual", "registered"]
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
        print(cat ,": siginifcant -> all population are not equal")
        print("------------")
    else:
        print(cat, ":not-significant -> all population are equal")
        print("------------")

# %%
## holiday x weather
holiday_weather = train.groupby(["holiday","weather"])["count"].mean().reset_index()
sns.pointplot(x = "holiday", y= "count", hue = 'weather', data = holiday_weather, join =False)

# %%
## workingday x weather
workingday_weather = train.groupby(["workingday","weather"])["count"].mean().reset_index()
sns.pointplot(x = "workingday", y= "count", hue = 'weather', data = workingday_weather, join =False)

# %%
## holiday x hour

holiday_hour = train.groupby(["holiday","hour"])["count"].mean().reset_index()
sns.pointplot(x = "hour", y= "count", hue = 'holiday', data = holiday_hour, join =True)

# %%
## workingday x hour
workingday_hour = train.groupby(["workingday","hour"])["count"].mean().reset_index()
sns.pointplot(x = "hour", y= "count", hue = 'workingday', data = workingday_hour, join =True)
   
# %%
## Visualization of numerical variable vs Target variable(Distribution)

corr_data = train[numerical_variable+ target_variable].corr()
print(corr_data)
sns.heatmap(corr_data, annot=True)


# %%
## Outlier analysis for Target variable

sns.distplot(train['count'], hist = True, color = 'blue')
plt.show()

# %% 
sns.boxplot(y ='count', data= train)
plt.show()

Q1 = train["count"].quantile(0.25)
Q3 = train["count"].quantile(0.75)
IQR = Q3 - Q1
Lower_Fence = Q1 - (1.5 * IQR)
Upper_Fence = Q3 + (1.5 * IQR)

print("num_of_outlier:", len(train.loc[train["count"]> Upper_Fence]))
print("num_of_data:", len(train))


# %%  Sample
dir(anova_lm(model))


