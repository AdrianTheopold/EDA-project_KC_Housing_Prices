### importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

### import data and convert data column entries to datatime object
df_house_complete =  pd.read_csv('kc_house_prices/King_County_House_prices_dataset.csv')
df_house_complete.date = pd.to_datetime(df_house_complete.date)
df_house_complete.sort_values('date',inplace=True)

# create some new features, age is easier to interpret as yr_built
df_house_complete.eval('price_per_sqft = price / sqft_living',inplace=True)
df_house_complete['month'] = df_house_complete['date'].dt.month
df_house_complete['year'] = df_house_complete['date'].dt.year
df_house_complete['age'] = 2015 - df_house_complete['yr_built']
df_house_complete.drop('yr_built',axis=1)

df_low_price = df_house_complete.query('price < 4000000')

df_house_complete['sqft_basement'] = df_low_price['sqft_basement'].replace('?','0').astype('float32')
df_house_complete['yr_renovated'].fillna(0,inplace=True)
df_house_complete['view'].fillna(0,inplace=True)
df_house_complete['waterfront'].fillna(0,inplace=True)

df_low_price = df_house_complete.query('price < 4000000')


age_cat = pd.get_dummies(pd.cut(df_low_price['age'],  bins=np.linspace(df_low_price['age'].min(), df_low_price['age'].max(), 20)), drop_first=True)
renovated_lastyr_cat = pd.get_dummies(pd.cut(df_low_price['yr_renovated'], bins = [0,2005,2020]), drop_first=True)
zipcodes_cat = pd.get_dummies(df_low_price['zipcode'], drop_first=True)


numerical_values = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','condition','grade','sqft_basement','sqft_living15','sqft_lot15','price_per_sqft','view']
categorical_values = ['waterfront','zipcode','yr_until_renovation']
predictors = ['bedrooms','bathrooms','sqft_living','sqft_lot','condition','grade','sqft_basement','sqft_living15','sqft_lot15','view','waterfront']

X = pd.concat([df_low_price[predictors],zipcodes_cat],axis=1)
X = pd.concat([X,age_cat],axis=1)
X = pd.concat([X,renovated_lastyr_cat],axis=1)

y = df_low_price['price']
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.OLS(y_train,X_train).fit()
print(model.summary())

### function for RMSE
def rmse(y_predict, y_target):
    return np.mean((y_predict - y_target) ** 2) ** 0.5

print('')
print('The RMSE for our test-set is:' + str(rmse(model.predict(X_test), y_test))) 