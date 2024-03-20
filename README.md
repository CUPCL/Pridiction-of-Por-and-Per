# Pridiction-of-Por-and-Per

Data Preprocessing provides pre-processing methods for logging data and measured data:
1. Convert txt files to Excel/CSV files in batches.
2. Merge the converted Excel/CSV file in pairwise and align the features.
3. Using well name and depth as index, the logging data of the same well location and the measured data are matched according to the closest depth.

Machine Learning provides all the Machine Learning model code used in this article: 
1. PI-Set includes PI-RF, PI-XGBoost, and PI-CatBoost.
2. ML-Without Stacking includes LR, SVR, KNN, DT, BPNN, RF, XGBoost, LightGBM, CatBoost.
3. ML-Stacking includes the Stacking model(Ensemble Learning).