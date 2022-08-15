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
fig, axs = plt.subplots(ncols=1, nrows=4, figsize=(20, 20))

for i, num in enumerate(numerical_variable):
    print(num)
    sns.distplot(train[num], label = num, hist = True, color = 'green', ax =axs[i])        
    axs[i].set_title("Distribution of {} Features".format(num), size = 20)   
    plt.subplots_adjust(top=1.5)
    axs[i].legend(loc = 'upper right')
    
plt.show()


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




''' Train '''
# %%

df_train = pd.read_csv("./Dataset/Regression/train.csv")
df_test = pd.read_csv("./Dataset/Regression/test.csv")

# %%
df_train.loc[df_train["datetime"]<'2012-06-01', "check"] = "train"
df_train.loc[df_train["datetime"]>= '2012-06-01',"check"] = "val"
df_test["check"] = "test"

# %%
df = pd.concat([df_train, df_test], axis = 0)
# %%  
def data_processing(data):
    
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["year"] = data["datetime"].dt.year
    data["month"] = data["datetime"].dt.month
    data["day"] = data["datetime"].dt.day
    data["hour"] = data["datetime"].dt.hour
    data["weekday"] = data["datetime"].dt.weekday ## 0: Mon ~ 6 : Sun
    
    data["season"] = data["season"].map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })
    data["weekday"] = data["weekday"].map({0: "Mon", 1:"Tue", 2:"Wen", 3:"Thr", 4:"Fri", 5:"Sat", 6:"Sun"})
    data["weather"] = data["weather"].map({1: "Clear + Few clouds",\
                                            2 : "Mist + Cloudy ", \
                                            3 : "Light Snow + Rain", \
                                            4 : "Heavy Rain + Ice Pallets" })
    
    
    categorical_variable = ["season", "workingday", "weather", "year", "month" ,"hour"]
    numerical_variable = ["temp", "atemp", "humidity"]
    check_variable = ["check"]
    target_variable =["count"]
    
    for cat in categorical_variable:
        data[cat] =data[cat].astype("category")
    
    X_data = pd.concat([data[numerical_variable+check_variable], pd.get_dummies(data[categorical_variable])], axis = 1)
    Y_data = data[target_variable+check_variable]
        
    x_train = X_data.loc[X_data["check"]=="train"]
    x_val = X_data.loc[X_data["check"]=="val"]
    x_test = X_data.loc[X_data["check"]=="test"]
    
    y_train = Y_data.loc[Y_data["check"]=="train"]
    y_val = Y_data.loc[Y_data["check"]=="val"]
    
    x_train = x_train.drop("check", axis =1)
    x_val = x_val.drop("check", axis =1)
    x_test = x_test.drop("check", axis =1)
    y_train = y_train.drop("check", axis =1)
    y_val = y_val.drop("check", axis =1)

        
    return x_train, y_train, x_val, y_val, x_test

# %%
X_train, y_train, X_val, y_val, X_test = data_processing(df)

# %%
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

random_state = 777

# %%
## RandomForestClassifier
reg_rf = RandomForestRegressor(random_state = random_state)


params_rf = { 'n_estimators' : [10, 50, 100],
           'max_depth' : [3, 6, 9],
           'min_samples_split' : [2, 5, 10]
            }

greg_rf = GridSearchCV(reg_rf,  param_grid = params_rf, cv = 5, n_jobs = -1)
greg_rf.fit(X_train.values, y_train.values.ravel())
rf_best = greg_rf.best_estimator_


#  %%
##GradientBoostingRegressor
reg_gb = GradientBoostingRegressor(random_state = random_state)


params_gb = { 'n_estimators' : [10, 50, 100],
           'max_depth' : [3, 6, 9],
           'min_samples_split' : [2, 5, 10]
            }

greg_gb = GridSearchCV(reg_gb,  param_grid = params_gb, cv = 5, n_jobs = -1)
greg_gb.fit(X_train.values, y_train.values.ravel())
gb_best = greg_gb.best_estimator_

# %%
y_pred_rf = rf_best.predict(X_val.values)
y_pred_gb = gb_best.predict(X_val.values)

# %%
y_pred_rf[np.where(y_pred_rf <0)] = 0
y_pred_gb[np.where(y_pred_gb <0)] = 0
# %%
print("rmse:", mean_squared_error(y_val.values, y_pred_rf)**0.5)
print("rmse:",mean_squared_error(y_val.values, y_pred_gb)**0.5)
# %%
print("rmsle:",mean_squared_log_error(y_val.values, y_pred_rf))
print("rmsle:",mean_squared_log_error(y_val.values, y_pred_gb))


# %%
