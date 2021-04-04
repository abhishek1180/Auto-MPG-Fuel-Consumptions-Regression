# %%
import pandas as pd
import numpy as np
from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


# %%
# Importing dataset
url="https://gist.githubusercontent.com/omarish/5687264/raw/7e5c814ce6ef33e25d5259c1fe79463c190800d9/mpg.csv"
dataframe=pd.read_csv(url)
dataframe.head()

# %%
# drop car_name as it is useless
dataframe1=dataframe.drop('name',axis=1)
dataframe1.head()

# %%
# converting number into dummy variables since origin 1,2,3 are continent names that are America, Europe, Asia respectively.
dataframe2=pd.get_dummies(dataframe1,columns=['origin'])
dataframe2.head()

# %%
# check missing values
print (dataframe2.dtypes, dataframe2.shape, sep='\n')

# %%
dataframe2.describe()

# %%
'''
No zero as minimum value in all columns except origin columns
'''

# %%
# Since there are missing values in horse power column by checking its dtype

dataframe2.horsepower.str.isdigit().sum()

# %%
#  we will convert replace nondigit with nan value
dataframe3=dataframe2.replace('?', np.nan)

# %%
null_columns=dataframe3.columns[dataframe3.isnull().any()]
print (null_columns)

# %%
'''
It means only horsepower column has missing values(NaN).
'''

# %%
# records with nan value
dataframe3[dataframe3["horsepower"].isnull()]

# %%
# replacing with median
dataframe3.median()
dataframe4=dataframe3.apply(lambda x: x.fillna(x.median()),axis=0)

# %%
dataframe4.dtypes

# %%
dataframe4['horsepower']=dataframe4['horsepower'].astype('float64')
dataframe4.dtypes

# %%
# building the correlation matrix
correlation_matrix = dataframe4.corr().round(2)
sns.heatmap(correlation_matrix, annot=True)

# %%
# splitting dataset into input and output
X=dataframe4.iloc[:,1:].values
y=dataframe4.iloc[:,0].values
print (X,y,sep='\n')

# %%
# splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)

# %%
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(X_train,y_train)

# %%
y_predict=linear_reg.predict(X_test)

# %%
linear_reg.intercept_

# %%
linear_reg.coef_

# %%
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)

# %%
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_predict)

# %%
from sklearn.model_selection import cross_val_score
scores=cross_val_score(linear_reg, X_train, y_train, cv=5)
print (scores, scores.mean(), sep='\n')

# %%
# Tuning hyperparameter
from sklearn.linear_model import RidgeCV

# %%
regression=RidgeCV(alphas=[0.5,0.1,0.05,2,3,4,5,6,7,8])

# %%
regression.fit(X_train,y_train)

# %%
regression.alpha_

# %%
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

for i in [0.05,0.5,0.1,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,23,24]:
    Ridge_regression=Ridge(alpha=i)
    Ridge_regression.fit(X_train,y_train)
    y_predicted=Ridge_regression.predict(X_test)
    print('Coefficients : ', Ridge_regression.coef_)
    print('r2_score : ', r2_score(y_test,y_predicted))
        