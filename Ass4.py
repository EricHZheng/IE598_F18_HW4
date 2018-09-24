#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 12:45:12 2018

@author: ericzheng
"""
#housing data https://raw.githubusercontent.com/rasbt/pythonmachine-learning-book-2nd-edition/master/code/ch10/housing.data.txt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

#print (df.head())
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 16)

#print head and tail of data frame
pd.set_option('display.width', 120)
print('Head and Tail:' + '\n')
print(df.head())
print(df.tail())
print('\n')

#1.1 print summary statistics of data frame
pd.set_option('display.width', 100)
print('Summary Statistics:' + '\n')
summary = df.describe()
print(summary)

#1.2 print box plots of data frame

sns.set(rc={'figure.figsize':(12,10)})
sns.boxplot(data=df.iloc[:,0:14])
print('\n' + 'Box Plots:')
plt.show()


for feature in df.columns:
    sns.set(rc={'figure.figsize':(5,3)})
    sns.boxplot( y=df[feature])
    plt.show()

#1.3 Scatterplot Matrix 
#cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
print('\n' + 'Scatterplot Matrix:')
cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()       
        
cols = ['NOX', 'RM', 'AGE', 'DIS', 'MEDV']        
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()

cols = ['RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'] 
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show() 
    
#1.3 Heatmap
corMat = pd.DataFrame(df.corr())
plt.pcolor(corMat)
print("\n" + "Heatmap:")
plt.tight_layout()
plt.show()

#1.4 Correlation Matrix 
print("\n" + "Correlation Matrix:")
cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
sns.set(rc={'figure.figsize':(12,9)})
#sns.set()
cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
# plt.savefig('images/10_04.png', dpi=300)
plt.show()

#1.5 Split data into training and test sets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


# Create training and test sets
X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT']].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)



#Part 2: Linear regression
#Normalizing the data
# =============================================================================
# scalerX = StandardScaler().fit(X_train)
# scalery = StandardScaler().fit(y_train)
# X_train = scalerX.transform(X_train)
# y_train = scalery.transform(y_train)
# X_test = scalerX.transform(X_test)
# y_test = scalery.transform(y_test)
# =============================================================================


# =============================================================================
# # Create the regressor: reg_all
# reg_all = LinearRegression()
# 
# # Fit the regressor to the training data
# reg_all.fit(X_train, y_train)
# 
# # Predict on the test data: y_pred
# y_pred = reg_all.predict(X_test)
# 
# # Compute and print R^2 and RMSE
# print("R^2: {}".format(reg_all.score(X_test, y_test)))
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("Root Mean Squared Error: {}".format(rmse))
# =============================================================================

sc_x = StandardScaler()
sc_y = StandardScaler()
X_train_std = sc_x.fit_transform(X_train)
y_train_std = sc_y.fit_transform(y_train[:, np.newaxis]).flatten()

# 2.1 Model Coefficients and y-intercept：
lr = LinearRegression()
lr.fit(X_train_std, y_train_std)
pd.set_option('display.max_rows', 16)
data = pd.Series(lr.coef_,index=[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']])
print("\n" + "Coefficients:")
print (data)
print("\n" + "y-intercept:")
print(lr.intercept_)


#2.2 plot the residual errors
print("\n" + "Residual errors:")
plt.figure(figsize=(8,6))
sc_x = StandardScaler()
sc_y = StandardScaler()
X_test_std = sc_x.fit_transform(X_test)
y_test_std = sc_y.fit_transform(y_test[:, np.newaxis]).flatten()

y_train_pred_std = lr.predict(X_train_std)
y_test_pred_std = lr.predict(X_test_std)

y_train_pred = sc_y.inverse_transform(y_train_pred_std)
y_test_pred = sc_y.inverse_transform(y_test_pred_std )

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

# plt.savefig('images/10_09.png', dpi=300)
plt.show()

#2.3 calculate performance metrics: MSE and R2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("\n")
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))



feature_names=['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT']
#Part 3.1: Ridge regression
print("\n"+'Ridge regression:')
from sklearn.linear_model import Ridge

ridge_df = pd.DataFrame()

for a in [0, 0.1, 1, 5]:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_std, y_train_std)
    y_train_pred_std = ridge.predict(X_train_std)
    y_test_pred_std = ridge.predict(X_test_std)
    
    array = {f: s for f, s in zip(feature_names, ridge.coef_)}
    array['alpha'] = a
    array['intercept'] = ridge.intercept_
    
    y_train_pred = sc_y.inverse_transform(y_train_pred_std)
    y_test_pred = sc_y.inverse_transform(y_test_pred_std )
    ###
    #plot residual errors
    #print('alpha=' + str(a))
    plt.clf()
    plt.figure(figsize=(8,6))
    plt.scatter(y_train_pred,  y_train_pred - y_train,
        c='steelblue', marker='o', edgecolor='white',
        label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values ' + '(alpha=' + str(a) + ')')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.tight_layout()
    plt.show()
    ###
    array['MSE test'] = mean_squared_error(y_test, y_test_pred)
    array['R^2 test'] = r2_score(y_test, y_test_pred)
    
    ridge_df=ridge_df.append(array, ignore_index=True)
    
ridge_df = ridge_df.set_index('alpha')[['MSE test','R^2 test'] + feature_names + ['intercept']]  
print("\n")
print(ridge_df)


    
#Part 3.2: LASSO regression
print("\n"+'Lasso regression:')
from sklearn.linear_model import Lasso

lasso_df = pd.DataFrame()

for a in [0, 0.01, 0.1, 1]:
    lasso = Lasso(alpha=a)
    lasso.fit(X_train_std, y_train_std)
    y_train_pred_std = lasso.predict(X_train_std)
    y_test_pred_std = lasso.predict(X_test_std)
    
    ###
    #plot residual errors
    #print('alpha=' + str(a))
    plt.clf()
    plt.figure(figsize=(8,6))
    plt.scatter(y_train_pred,  y_train_pred - y_train,
        c='steelblue', marker='o', edgecolor='white',
        label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values ' + '(alpha=' + str(a) + ')')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.tight_layout()
    plt.show()
    ###
    
    array = {f: s for f, s in zip(feature_names, lasso.coef_)}
    array['alpha'] = a
    #print (a)
    array['intercept'] = lasso.intercept_
    
    y_train_pred = sc_y.inverse_transform(y_train_pred_std)
    y_test_pred = sc_y.inverse_transform(y_test_pred_std )
    
    array['MSE test'] = mean_squared_error(y_test, y_test_pred)
    array['R^2 test'] = r2_score(y_test, y_test_pred)
    
    lasso_df=lasso_df.append(array, ignore_index=True)
    
lasso_df = lasso_df.set_index('alpha')[['MSE test','R^2 test'] + feature_names + ['intercept']]  
print("\n")
print(lasso_df)

#Part 3.3: Elastic Net regression
print("\n"+'ElasticNet regression:')
from sklearn.linear_model import ElasticNet

elanet_df = pd.DataFrame()

for a in [0, 0.01, 0.1, 0.5, 1]:
    elanet = ElasticNet(alpha=1.0, l1_ratio=a)
    elanet.fit(X_train_std, y_train_std)
    y_train_pred_std = elanet.predict(X_train_std)
    y_test_pred_std = elanet.predict(X_test_std)
    
    ###
    #plot residual errors
    #print('alpha=' + str(a))
    plt.clf()
    plt.figure(figsize=(8,6))
    plt.scatter(y_train_pred,  y_train_pred - y_train,
        c='steelblue', marker='o', edgecolor='white',
        label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values ' + '(l1_ratio=' + str(a) + ')')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.tight_layout()
    plt.show()
    ###
    
    array = {f: s for f, s in zip(feature_names, elanet.coef_)}
    array['l1_ratio'] = a
    #print (a)
    array['intercept'] = elanet.intercept_
    
    y_train_pred = sc_y.inverse_transform(y_train_pred_std)
    y_test_pred = sc_y.inverse_transform(y_test_pred_std )
    
    array['MSE test'] = mean_squared_error(y_test, y_test_pred)
    array['R^2 test'] = r2_score(y_test, y_test_pred)
    
    elanet_df=elanet_df.append(array, ignore_index=True)
    
elanet_df = elanet_df.set_index('l1_ratio')[['MSE test','R^2 test'] + feature_names + ['intercept']]  
print("\n")
print(elanet_df)

print("My name is Hao Zheng")
print("My NetID is: haoz7")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")