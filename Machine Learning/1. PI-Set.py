import eli5
import numpy as np
import pandas as pd
from matplotlib import rcParams
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

'''PI-Set'''

# Global font setting
config = {
            "font.family": 'serif',
            "mathtext.fontset": 'stix',
            "font.serif": ['Times New Roman'],
         }
rcParams.update(config)

# Load data set
dataset = pd.read_excel('File path.xlsx')
X = dataset.iloc[0:, 0:7].values
y = dataset.iloc[0:, 7].values
#y = np.log(y) #The permeability needs to be logged

# Read feature name
feature_names = dataset.columns.tolist()

# Partition data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# DT
my_model1 = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)

# PI-DT
perm1 = PermutationImportance(my_model1, random_state=42).fit(X_test, y_test)

# Show
eli5.show_weights(perm1, feature_names = ['Depth', 'SP', 'GR', 'CAL', 'CNL', 'DEN', 'AC'], top = None)

# RF
my_model2 = RandomForestRegressor(random_state=42).fit(X_train, y_train)

# PI-RF
perm2 = PermutationImportance(my_model2, random_state=42).fit(X_test, y_test)

# Show
eli5.show_weights(perm2, feature_names = ['Depth', 'SP', 'GR', 'CAL', 'CNL', 'DEN', 'AC'], top = None)

# XGBoost
my_model3 = XGBRegressor(random_state=42).fit(X_train, y_train)

# PI-XGBoost
perm3 = PermutationImportance(my_model3, random_state=42).fit(X_test, y_test)

# Show
eli5.show_weights(perm3, feature_names = ['Depth', 'SP', 'GR', 'CAL', 'CNL', 'DEN', 'AC'], top = None)

# CatBoost
my_model4 = CatBoostRegressor(random_state=42).fit(X_train, y_train)

# PI-CatBoost
perm4 = PermutationImportance(my_model4, random_state=42).fit(X_test, y_test)

# Show
eli5.show_weights(perm4, feature_names = ['Depth', 'SP', 'GR', 'CAL', 'CNL', 'DEN', 'AC'], top = None)