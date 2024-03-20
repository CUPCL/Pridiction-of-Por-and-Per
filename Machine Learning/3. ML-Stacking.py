import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

config = {
            "font.family": 'serif',
            "font.size": 15,
            "mathtext.fontset": 'stix',
            "font.serif": ['Times New Roman'],
         }
rcParams.update(config)

# Load data set
data = pd.read_excel('File Path')
print(data)
print(data.describe())
X = data.iloc[0:, 0:6].values
#print(X)
y = data.iloc[0:, 6].values
#y = np.log(y)

# Data preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=20)

# Define a base learner and a meta-learner
base_learners = [
    ('rf1', RandomForestRegressor(n_estimators=222, max_depth = 15, random_state=42)),
    ('rf2', RandomForestRegressor(random_state=35)),
    ('dt', DecisionTreeRegressor()),
    ('cb1', CatBoostRegressor(iterations=535, depth = 9, learning_rate = 0.01325, random_state=32)),
    ('cb2', CatBoostRegressor(iterations=600, random_state=42)),
    ('lgb', LGBMRegressor(learning_rate = 0.037,max_depth = 9)),
#    ('mlp', MLPRegressor()),
#    ('svr', SVR()),
    ('knn', KNeighborsRegressor()),
    ('xgb', XGBRegressor(n_estimators = 924, learning_rate = 0.06)),
]
meta_learner = LinearRegression()

# Create pipeline flow
stacking_regressor = make_pipeline(StackingRegressor(estimators=base_learners, final_estimator=meta_learner))

# Train the model and predict
stacking_regressor.fit(X_train, y_train)
y_pred = stacking_regressor.predict(X_test)
score = stacking_regressor.score(X_test,y_test)

# Calculation and evaluation metrix
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Mean Absolute Percentage Error

plt.figure()
plt.plot(np.arange(len(y_pred)), y_test, 'go-', label = 'true value')
plt.plot(np.arange(len(y_pred)), y_pred, 'ro-', label = 'predict value')
plt.title('Stacking: %f'%score)
plt.legend()
plt.show()

print(f'R^2: {r2:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.4f}')