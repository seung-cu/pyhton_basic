###### Importing Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
import warnings
warnings.filterwarnings('ignore')

os.chdir("C:/Users/user/Desktop/승모/이직/Codility/DataSet/Classification")
os.listdir()

###### Loading Data set

train = pd.read_csv("./train.csv")
test= pd.read_csv("./test.csv")
submission = pd.read_csv("./gender_submission.csv")

###### Checking Data set

print("length of data:", len(train))
print(train.head())
print(train.info())
print(len(train[train["Survived"]==1]))
print(train.isnull().sum())
print(test.isnull().sum())


###### Exploratory Data Analysis (EDA)

#### Handling Missing Value
print(train.isnull().sum())
print(test.isnull().sum())

## concatenating train, test data set
train["check"] ="train"
test["check"] = "test"
df_all = pd.concat([train, test], axis = 0)
print(df_all.isnull().sum())

## Cabin -> Dropping this columns becasue of too many null value
df_all = df_all.drop("Cabin", axis =1)

## Embarked -> Filling S(The most common value)
df_all.Embarked.value_counts()
df_all.loc[df_all["Embarked"].isnull(), "Embarked"] = 'S'

## Age -> too many null value, howevere age seems to be promising for determining survival rate -> Filling NA based on Pcalss, Sex
df_all.corr()
df_all['Age'] = df_all["Age"].fillna(df_all.groupby(['Sex', 'Pclass'])['Age'].transform("median"))

## Fare -> Using Mean
df_all.loc[df_all["Fare"].isnull(), "Fare"] = df_all["Fare"].mean()
print(df_all.isnull().sum())

## splitting train, test data set
train = df_all.loc[df_all["check"]=="train"].drop("check", axis=1)
test = df_all.loc[df_all["check"]=="test"].drop(["check", "Survived"], axis=1)

## Categorizing data types

'''
PassengerId - this is a just a generated Id(string)
Pclass - which class did the passenger ride - first, second or third (Categorical)
Name - self explanatory (string)
Sex - male or female (Categorical)
Age (Numeric)
SibSp - were the passenger's spouse or siblings with them on the ship (Categorical)
Parch - were the passenger's parents or children with them on the ship (Categorical)
Ticket - ticket number (string)
Fare - ticker price (Numeric)
Cabin (string)
Embarked - port of embarkation (Categorical)
Survived - did the passenger survive the sinking of the Titanic (Target, Categorical)
'''

string_variable = ["PassengerId", "Name", "Ticket"]
categorical_variable = ["Pclass", "Sex", "SibSp","Parch","Embarked"]
numeric_variable = ["Age", "Fare"]
target_variable =["Survived"]

train[string_variable] = train[string_variable].astype(str)
train[categorical_variable] = train[categorical_variable].astype(object)


#### Which feature has relationship with target variable? 

### Categorical Variable - Which categorical variable have an effect on target variable?
## All categorical variables have posibillity to have an effect on target variable
## But Pclass and Sex look like best categorical features becasue they have the significant difference between groups. 
fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))

for i, cat in enumerate(categorical_variable):
    print(cat)
    temp = train.groupby([cat,"Survived"])["PassengerId"].count().reset_index()
    temp = temp.assign(passenger_sum = temp.groupby(cat)["PassengerId"].transform('sum')) ## Window function
    temp["survived_ratio"] = temp["PassengerId"]/temp["passenger_sum"]*100
    print(temp.loc[temp["Survived"]==1][[cat, "Survived", "survived_ratio"]])
    
    # train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    
    ## Drop Null value for visualization
    train_temp = train[[cat, "Survived"]].dropna()
    plt.subplot(2, 3, i+1)
    sns.countplot(x=cat, hue='Survived', data=train_temp)
    
    plt.xlabel('{}'.format(cat), size=20, labelpad=15)
    plt.ylabel('Passenger Count', size=20, labelpad=15)    
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    
    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})
    plt.title('Count of Survival in {} Feature'.format(cat), size=20, y=1.05)

plt.show()


### Numeric Variable - Which nemeric variable have a strong relationship with target variable?
## Fare is better variable than Age. Because Age has no the significant difference between Survived and Not_Survived group
## Very young age passanger were more survived

fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(20, 20))

for i, num in enumerate(numeric_variable):
    print(num)
    print(train[[num, "Survived"]].corr())
    
    sns.distplot(train.loc[train["Survived"]==1][num], label = "Survived", hist = True, color = 'blue', ax =axs[i])
    sns.distplot(train.loc[train["Survived"]==0][num], label = "Not_Survived", hist = True, color = 'red', ax =axs[i])
        
    axs[i].set_title("Distribution of {} Features".format(num), size = 20)   
    plt.subplots_adjust(top=0.8)
    axs[i].legend(loc = 'upper right')
    
plt.show()

### Combined Feature Relations(Categoricla and Numerical Variable)
for cat in categorical_variable:
    print(cat)
    temp = train.loc[train[cat].isnull()==False]
    for num in numeric_variable:    
        temp = temp.loc[temp[num].isnull() == False]
        pal = {1:"blue", 0:"red"}
        g = sns.FacetGrid(temp, height=5, col=cat, margin_titles=True, hue = "Survived",
                          palette=pal)
        g = g.map(plt.hist, num, alpha = 0.5).add_legend();
        g.fig.suptitle("Survived by {} and {}".format(cat, num), size = 15)
        plt.subplots_adjust(top=0.8)
        plt.show()


##### Feature Engineering
from sklearn.preprocessing import OneHotEncoder, StandardScaler

### Age -> splitting into three grouop
def make_age_group(row):
    if row < 10:
        value = "child"
    elif row > 60:
        value = "old"
    else :
        value ="young"
    
    return value
     
train["Age_Group"] = train.apply(lambda x : make_age_group(x["Age"]), axis = 1)
test["Age_Group"] = test.apply(lambda x : make_age_group(x["Age"]), axis = 1)
categorical_variable.append("Age_Group")


### One-Hot Encoding of Categorical Variable
for cat in categorical_variable:    
    train = pd.get_dummies(train, prefix = [cat], columns = [cat], drop_first=False)
    test = pd.get_dummies(test, prefix = [cat], columns = [cat], drop_first=False)
    

### Scailing of Numerical Variable
scaler = StandardScaler()
scaler.fit(pd.concat([train["Fare"],test["Fare"]], axis=0))
train[["scaled_fare"]] = scaler.transform(train["Fare"])
test[["scaled_fare"]] =  scaler.transform(test["Fare"])


print(train.columns)

### Making Train, Test Data Set
X_train = train[['Pclass_1','Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1','SibSp_2', \
                 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4',\
                 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Age_Group_child', 'Age_Group_old', \
                 'Age_Group_young', 'scaled_fare']].values
    
X_test = test[['Pclass_1','Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1','SibSp_2', \
                 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4',\
                 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Age_Group_child', 'Age_Group_old', \
                 'Age_Group_young', 'scaled_fare']].values

Y_train = train["Survived"].values


print('X_train shape: {}'.format(X_train.shape))
print('Y_train shape: {}'.format(Y_train.shape))
print('X_test shape: {}'.format(X_test.shape))



##### Training

### Testing differents algorithms
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


random_state= 42
classifiers = []

classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(DecisionTreeClassifier(random_state = random_state))
classifiers.append(RandomForestClassifier(random_state = random_state))
classifiers.append(GradientBoostingClassifier(random_state = random_state))
classifiers.append(SVC(random_state = random_state))
classifiers.append(KNeighborsClassifier())

cv_results = []
for classifier in classifiers :
    print(classifier)
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = 5, n_jobs=-1))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())


cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["LogisticRegression", "DecisionTreeClassifier"\
                                                                                      ,"RandomForestClassifier", "GradientBoostingClassifier", "SVC","KNeighborsClassifier"]})



print(cv_res.sort_values("CrossValMeans", ascending = False))    

g = sns.barplot( "CrossValMeans","Algorithm", data = cv_res, orient = "h")
g = g.set_title("Cross validation scores")




### Parmeter tuning with SVC, GradientBoostingClassifier, KNeighborsClassifier, RandomForestClassifier

from sklearn.model_selection import GridSearchCV

grid_serch_result = {}

## RandomForestClassifier
clf_rf = RandomForestClassifier(random_state = random_state)


params_rf = { 'n_estimators' : [10, 50, 100],
           'max_depth' : [3, 6, 9],
           'min_samples_split' : [2, 5, 10]
            }

gclf_rf = GridSearchCV(clf_rf,  param_grid = params_rf, cv = 5, n_jobs = -1)
gclf_rf.fit(X_train, Y_train)
rf_best = gclf_rf.best_estimator_



## GradientBoostingClassifier
clf_gbc = GradientBoostingClassifier(random_state = random_state)

params_gbc = {
     "n_estimators": [50, 100, 150],
     "max_depth": [3, 6, 9],
     "min_samples_split" : [2, 5, 10],
     "learning_rate": [0.001, 0.01, 0.1]
    }


gclf_gbc = GridSearchCV(clf_gbc,  param_grid = params_gbc, cv = 5, n_jobs = -1)
gclf_gbc.fit(X_train, Y_train)
gbc_best = gclf_gbc.best_estimator_


## Support Vector Classifier
clf_svc = SVC(random_state = random_state)

params_svc = {
    'C': [0.1,1, 10, 100], ## lower c, smaller error -> increase over fitting
    'gamma': [1,0.1,0.01,0.001], ## Gamma high means more curvature when using kernel
    'kernel': ['rbf', 'poly', 'sigmoid']
    }


gclf_svc = GridSearchCV(clf_svc,  param_grid = params_svc, cv = 5, n_jobs = -1)
gclf_svc.fit(X_train, Y_train)
svc_best = gclf_svc.best_estimator_



## KNeighborsClassifier
clf_knc = KNeighborsClassifier()

params_knc = {
    'n_neighbors': [2, 5, 10], ## lower c, smaller error -> increase over fitting
    'weights':['uniform', 'distance'], ## Gamma high means more curvature when using kernel
    }

gclf_knc = GridSearchCV(clf_knc,  param_grid = params_knc, cv = 5, n_jobs = -1)
gclf_knc.fit(X_train, Y_train)
knc_best = gclf_knc.best_estimator_

print("-RandomForestClassifier:","\n After tuning :",gclf_rf.best_score_, "\n Before tuning :",cv_res.loc[cv_res["Algorithm"]=="RandomForestClassifier"]["CrossValMeans"].values[0])
print("-GradientBoostingClassifier:","\n After tuning :",gclf_gbc.best_score_, "\n Before tuning :",cv_res.loc[cv_res["Algorithm"]=="GradientBoostingClassifier"]["CrossValMeans"].values[0])
print("-SVC:","\n After tuning :",gclf_svc.best_score_, "\n Before tuning :",cv_res.loc[cv_res["Algorithm"]=="SVC"]["CrossValMeans"].values[0])
print("-KNeighborsClassifier:","\n After tuning :",gclf_knc.best_score_, "\n Before tuning :",cv_res.loc[cv_res["Algorithm"]=="KNeighborsClassifier"]["CrossValMeans"].values[0])



##### Model Result Analysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, balanced_accuracy_score

y_pred = gbc_best.predict(X_train)


# printing confision matrix
pd.DataFrame(confusion_matrix(Y_train,y_pred),\
            columns=["Predicted Not-Survived", "Predicted Survived"],\
            index=["Not-Survived","Survived"] )

accuracy_score(Y_train,y_pred)
recall_score(Y_train,y_pred)
print(classification_report(Y_train,y_pred))


## ROC Curve, AUC
from sklearn.metrics import roc_curve, roc_auc_score
proba = gbc_best.predict_proba(X_train)
fper, tper, thresholds = roc_curve(Y_train,proba[:,1])

plt.plot(fper, tper, color='red', label='ROC')
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend()
plt.show()

roc_auc_score(Y_train, y_pred)


### Submission, Prediciton using test data set
y_pred_test = gbc_best.predict(X_test)

submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission_df['PassengerId'] = test['PassengerId']
submission_df['Survived'] = y_pred_test
