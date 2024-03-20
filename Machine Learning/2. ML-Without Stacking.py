import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

config = {
            "font.family": 'serif',
            "font.size": 15,
            "mathtext.fontset": 'stix',
            "font.serif": ['Times New Roman'],
         }
rcParams.update(config)

dataset = pd.read_excel('File Path')
print(dataset)
print(dataset.describe())


X = dataset.iloc[0:, 0:7].values
y = dataset.iloc[0:, 7].values
#y = np.log(y)
print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 20)

# boxplot
column = dataset.columns.tolist()
fig = plt.figure(figsize = (12,4), dpi = 128)
for i in range(8):
    plt.subplot(2,4, i + 1)
    sns.boxplot(data = dataset[column[i]], orient = "v",width = 0.5)
    plt.ylabel(column[i], fontsize = 12)
plt.tight_layout()
plt.show()

for col in dataset.columns:
    q1 = dataset[col].quantile(0.25)
    q3 = dataset[col].quantile(0.75)
    iqr = q3 - q1
    outliers = dataset[(dataset[col] < q1 - 1.5 * iqr) | (dataset[col] > q3 + 1.5 * iqr)][col]
    if not outliers.empty:
        print(f'outlier of {col}：\n{outliers}\n')

# kdeplot
column = dataset.columns.tolist()
fig = plt.figure(figsize = (12,4), dpi = 128)
for i in range(8):
    plt.subplot(2,4, i + 1)
    sns.kdeplot(data=dataset[column[i]],color = 'blue', fill = True)
    plt.ylabel(column[i], fontsize = 12)
plt.tight_layout()
plt.show()

# Scatter plot
sns.pairplot(dataset[column],diag_kind='kde')
plt.tight_layout()
plt.savefig('Scatter plot.jpg',dpi=256)


# Pearson's correlation coefficient heatmap
corr1 = plt.figure(figsize = (10,10),dpi=128)
corr1 = sns.heatmap(dataset.corr(method = 'pearson'),annot = True, square = True)
plt.xticks(rotation = 45)
plt.title('Pearson heat map')
plt.show()

# Spearman's correlation coefficient heatmap
corr2 = plt.figure(figsize = (10,10),dpi=128)
corr2 = sns.heatmap(dataset.corr(method = 'spearman'),annot = True, square = True)
plt.xticks(rotation = 45)
plt.title('Spearman heat map')
plt.show()

# data normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

'''Tree-based algorithm'''
# Decision Tree
from sklearn.tree import DecisionTreeRegressor
DT_Reg = DecisionTreeRegressor(random_state = 42).fit(X_train, y_train)
y_pred = DT_Reg.predict(X_test)
score = DT_Reg.score(X_test,y_test)
r2 = r2_score(y_true = y_test, y_pred = y_pred)
r2_cv = cross_val_score(DT_Reg, X_test, y_test, cv = 2, scoring = "r2").mean()
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)
mape = mean_absolute_percentage_error(y_test,y_pred)
result = DT_Reg.predict(X_test)
print('DT-r2', r2)
print('DT-r2_cv', r2_cv)
print('DT-MSE', mse)
print('DT-RMSE', rmse)
print('DT-MAE', mae)
print('DT-MAPE', mape)
plt.figure()
plt.plot(np.arange(len(result)), y_test, 'go-', label = 'true value')
plt.plot(np.arange(len(result)), result, 'ro-', label = 'predict value')
plt.title('DTscore: %f'%score)
plt.legend()
plt.show()
print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

# Random Forest
from sklearn.ensemble import RandomForestRegressor

RF_Reg = RandomForestRegressor(random_state = 42)
RF_Reg.fit(X_train, y_train)
y_pred = RF_Reg.predict(X_test)
score = RF_Reg.score(X_test,y_test)
r2 = r2_score(y_true = y_test, y_pred = y_pred)
r2_cv = cross_val_score(RF_Reg,X_test,y_test,cv = 2, scoring = "r2").mean()
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)
mape = mean_absolute_percentage_error(y_test,y_pred)
result = RF_Reg.predict(X_test)

print('RF-r2', r2)
print('RF-r2_cv', r2_cv)
print('RF-MSE', mse)
print('RF-RMSE', rmse)
print('RF-MAE', mae)
print('RF-MAPE', mape)
print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

plt.figure()
plt.plot(np.arange(len(result)), y_test, 'go-', label = 'true value')
plt.plot(np.arange(len(result)), result, 'ro-', label = 'predict value')
plt.title('RFscore: %f'%score)
plt.legend()
plt.show()

# XGBoost
from xgboost import XGBRegressor
XGB_Reg = XGBRegressor(random_state = 42).fit(X_train, y_train)
y_pred = XGB_Reg.predict(X_test)
score = XGB_Reg.score(X_test,y_test)
r2 = r2_score(y_true = y_test, y_pred = y_pred)
r2_cv = cross_val_score(XGB_Reg,X_test,y_test, cv = 2, scoring = "r2").mean()
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)
mape = mean_absolute_percentage_error(y_test,y_pred)
result = XGB_Reg.predict(X_test)

print('XGBoost-r2', r2)
print('XGBoost-r2_cv', r2_cv)
print('XGBoost-MSE', mse)
print('XGBoost-RMSE', rmse)
print('XGBoost-MAE', mae)
print('XGBoost-MAPE', mape)
print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

plt.figure()
plt.plot(np.arange(len(result)), y_test, 'go-', label = 'true value')
plt.plot(np.arange(len(result)), result, 'ro-', label = 'predict value')
plt.title('XGBoostscore: %f'%score)
plt.legend()
plt.show()

# LightGBM
import lightgbm as lgb
LightGBMReg = lgb.LGBMRegressor(objective='regression',
                                boost_from_average = True, random_state = 42).fit(X_train, y_train)
y_pred = LightGBMReg.predict(X_test)
score = LightGBMReg.score(X_test,y_test)
r2 = r2_score(y_true = y_test, y_pred = y_pred)
r2_cv = cross_val_score(LightGBMReg, X_test, y_test,cv = 2, scoring = "r2").mean()
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)
mape = mean_absolute_percentage_error(y_test,y_pred)
result = LightGBMReg.predict(X_test)

print('LightGBM-r2', r2)
print('LightGBM-r2_cv', r2_cv)
print('LightGBM-MSE', mse)
print('LightGBM-RMSE', rmse)
print('LightGBM-MAE', mae)
print('LightGBM-MAPE', mape)
print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

plt.figure()
plt.plot(np.arange(len(result)), y_test, 'go-', label = 'true value')
plt.plot(np.arange(len(result)), result, 'ro-', label = 'predict value')
plt.title('LightGBMscore: %f'%score)
plt.legend()
plt.show()

from catboost import CatBoostRegressor
CatBoostReg = CatBoostRegressor().fit(X_train, y_train)
y_pred = CatBoostReg.predict(X_test)
score = CatBoostReg.score(X_test,y_test)
r2 = r2_score(y_true = y_test, y_pred = y_pred)
r2_cv = cross_val_score(CatBoostReg,X_test,y_test,cv = 2, scoring = "r2").mean()
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)
mape = mean_absolute_percentage_error(y_test,y_pred)
result = CatBoostReg.predict(X_test)

print('CatBoostReg-r2', r2)
print('CatBoostReg-r2_cv', r2_cv)
print('CatBoostReg-MSE', mse)
print('CatBoostReg-RMSE', rmse)
print('CatBoostReg-MAE', mae)
print('CatBoostReg-MAPE', mape)
print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

plt.figure()
plt.plot(np.arange(len(result)), y_test, 'go-', label = 'true value')
plt.plot(np.arange(len(result)), result, 'ro-', label = 'predict value')
plt.title('CatBoostRegscore: %f'%score)
plt.legend()
plt.show()

'''常规机器学习模型'''
# Linear Regression
LR = linear_model.LinearRegression()
LR.fit(X_train,y_train)
y_pred = LR.predict(X_test)
score = LR.score(X_test,y_test)
r2 = r2_score(y_true = y_test, y_pred = y_pred)
r2_cv = cross_val_score(LR,X_test, y_test,cv = 2, scoring="r2").mean()
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)
mape = mean_absolute_percentage_error(y_test,y_pred)
result = LR.predict(X_test)

print('LR-r2', r2)
print('LR-r2_cv', r2_cv)
print('LR-MSE', mse)
print('LR-RMSE', rmse)
print('LR-MAE', mae)
print('LR-MAPE', mape)
print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

plt.figure()
plt.plot(np.arange(len(result)), y_test,'go-',label = 'true value')
plt.plot(np.arange(len(result)),result,'ro-',label = 'predict value')
plt.title('LRscore: %f'%score)
plt.legend()
plt.show()

# SVR
from sklearn import svm
SVR = svm.SVR()
SVR.fit(X_train,y_train)
score = SVR.score(X_test, y_test)
y_pred = SVR.predict(X_test)
result = SVR.predict(X_test)
r2 = r2_score(y_true = y_test, y_pred = y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)
mape = mean_absolute_percentage_error(y_test,y_pred)
result = SVR.predict(X_test)

print('SVR-r2', r2)
print('SVR-r2_cv', r2_cv)
print('SVR-MSE', mse)
print('SVR-RMSE', rmse)
print('SVR-MAE', mae)
print('SVR-MAPE', mape)
print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

plt.figure()
plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
plt.title('SVRscore: %f'%score)
plt.legend()
plt.show()

# KNN
from sklearn import neighbors
KNN = neighbors.KNeighborsRegressor()
KNN.fit(X_train,y_train)
score = KNN.score(X_test, y_test)
y_pred = KNN.predict(X_test)
result = KNN.predict(X_test)
score = KNN.score(X_test, y_test)
y_pred = KNN.predict(X_test)
result = KNN.predict(X_test)
r2 = r2_score(y_true = y_test, y_pred = y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)
mape = mean_absolute_percentage_error(y_test,y_pred)
result = KNN.predict(X_test)
print('KNN-r2', r2)
print('KNN-r2_cv', r2_cv)
print('KNN-MSE', mse)
print('KNN-RMSE', rmse)
print('KNN-MAE', mae)
print('KNN-MAPE', mape)
print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

plt.figure()
plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
plt.title('KNNscore: %f'%score)
plt.legend()
plt.show()

'''Deep learning'''

# BP neural network
from sklearn.neural_network import MLPRegressor

# Define a neural network model and train it
# hidden_layer_sizes: Number of hidden layer nodes; max_iter: indicates the maximum number of iterations
BP_Reg = MLPRegressor(hidden_layer_sizes=(5, 2), max_iter=10000).fit(X_train, y_train)
y_pred = BP_Reg.predict(X_test)
score = BP_Reg.score(X_test,y_test)
r2 = r2_score(y_true = y_test, y_pred = y_pred)
r2_cv = cross_val_score(RF_Reg,X_test, y_test,cv = 2, scoring="r2").mean()
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)
mape = mean_absolute_percentage_error(y_test,y_pred)
result = RF_Reg.predict(X_test)

print('BP-r2', r2)
print('BP-r2_cv', r2_cv)
print('BP-MSE', mse)
print('BP-RMSE', rmse)
print('BP-MAE', mae)
print('BP-MAPE', mape)
print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

plt.figure()
plt.plot(np.arange(len(result)), y_test, 'go-', label = 'true value')
plt.plot(np.arange(len(result)), result, 'ro-', label = 'predict value')
plt.title('BPscore: %f'%score)
plt.legend()
plt.show()