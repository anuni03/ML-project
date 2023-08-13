import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
housing = pd.read_csv("data.csv")
# print(housing.head()) --> starting table
# print(housing.info()) --> to see for missing data
# print(housing['CHAS'].value_counts()) --> how many 0s and 1s
# print(housing.describe()) --->gives various values mean,std,max,min etc
#housing.hist(bins=50, figsize=(20,15)) --> to create a histogram
#plt.show()
# Train-Test Spliting


"""def split_train_test(data, test_ratio): -->for learning
    np.random.seed(42)   # to fix the suffle value so that our model don't see all values
    shuffled=np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

train_set,test_set= split_train_test(housing,0.2)

print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}") """

train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
#   print(strat_test_set['CHAS'].value_counts()) #95/7=376/28
#   print(strat_train_set['CHAS'].value_counts()) # same ration of 0s and 1s
housing=strat_train_set.copy()

#Looking for correlations
non_numeric_columns = housing.select_dtypes(exclude=['float64', 'int64']).columns
label_encoder = LabelEncoder()   ## Example of label encoding (replace categorical values with integers)
for column in non_numeric_columns:
    housing[column] = label_encoder.fit_transform(housing[column])
corr_matrix = housing.corr(method='pearson')
# 1- strong positive correlation(more value increses the label)

print(corr_matrix['MEDV'].sort_values(ascending=False)) 
attribes=["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attribes],figsize=(12,8))
plt.show()
housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)
plt.show()

# Trying out Attribute combinations can create correlation as above 
housing["TAXRM"]=housing["TAX"]/housing["RM"]
print(housing.head())

housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set["MEDV"].copy()

""" To take care of missing attributes we have 3 options
  1-Get rid of missing data points (less in number)
  2-Get rid of the whole attribute (corelation pointing to zero)
  3-Set the value to some value(0,mean or median) """

#a=housing.dropna(subset=["RM"]) #option 1
#a.shape ---------Original dataframe unchanged in all 3
# housing.drop("RM",axis=1).shape #Option 2 there is no RM column 
# median=housing["RM"].median()
# housing["RM"].fillna(median)  # Option 3

# Directly doing above task through sklearn
#imputer = SimpleImputer(strategy ="median")
#imputer.fit(housing)
# to see ---> imputer.statistics_

# X=imputer.transform(housing)

#Scikit-learn Design
""" Primarily , three types of objects
1. Estimators - It estimates some parameter based on a dataset. Eg-imputer --It has a fit method and transform method.
Fit method - fits the dataset and calculates internal parameters
2. Transformers - transform method takes input and returns output based on the learnings from fit(). It alse have a convenience function called fit_transform() which fits and then transforms.
3. Predictors  - LinearRegression model is a example of predictor. fit() and predict() are two common functions
It also gives score function which will evaluate the predictions."""

"""Feature Scaling 
Primarily, two types of features scaling
1. Min-max Scaling(Normalization)
(value-min)/(max-min)
Sklearn provides a class called MinMaxSCaler for this
2-Standardization
(value-mean)/std
Sklearn provides a class called StandardScaler for this 
"""
## CREATING A PIPELINE
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    # ......add as many as you want in your pipeline
    ('std_scaler',StandardScaler()),
])
housing_num_tr=my_pipeline.fit_transform(housing)
# print(housing_num_tr) --gives values between 0 and 1
print(housing_num_tr.shape)


# Selecting a desired model for Dragon Real Estate
#model=LinearRegression()
#model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)

some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
print(model.predict(prepared_data))
print(some_labels)

#Evaluating the model
housing_predictions=model.predict(housing_num_tr)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)

#Using better evaluation technique--Cross Validation
scores=cross_val_score(model,housing_num_tr,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
def print_scores(scores):
    print("Scores",scores)
    print("Mean",scores.mean())
    print("Standard Deviation",scores.std())

print_scores(rmse_scores)

#Saving the model
dump(model,'Dragon.joblib')

#Testing the model
X_test=strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set["MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
