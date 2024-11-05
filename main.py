import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cleaner
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error,  root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm

dataset = pd.read_csv("data/properties.csv")
''''   [id', 'price', 'property_type', 'subproperty_type', 'region',
       'province', 'locality', 'zip_code', 'latitude', 'longitude',
       'construction_year', 'total_area_sqm', 'surface_land_sqm',
       'nbr_frontages', 'nbr_bedrooms', 'equipped_kitchen', 'fl_furnished',
       'fl_open_fire', 'fl_terrace', 'terrace_sqm', 'fl_garden', 'garden_sqm',
       'fl_swimming_pool', 'fl_floodzone', 'state_building',
       'primary_energy_consumption_sqm', 'epc', 'heating_type',
       'fl_double_glazing', 'cadastral_income']
'''

###################### CLEAN UP DATASET - START ######################
# Step 1: Clean the dataset by removing columns with too many missing values
ds1 = cleaner.clean_dataset(dataset, missing_threshold=0.3, retain_columns=['surface_land_sqm'])
''''
    ['id', 'price', 'property_type', 'subproperty_type', 'region',
    'province', 'locality', 'zip_code', 'latitude', 'longitude',
    'total_area_sqm', 'nbr_bedrooms', 'fl_furnished', 'fl_open_fire',surface_land_sqm,
    'fl_terrace', 'terrace_sqm', 'fl_garden', 'garden_sqm',
    'fl_swimming_pool', 'fl_floodzone', 'fl_double_glazing']
'''

# Step 2: Replace NaN values with mean values for specific columns
# ds2 = cleaner.remove_NaN(ds1)
ds2 = ds1
# Step 2.1: Manualy remove columns whe don't need
ds2.drop(['id', 'subproperty_type', 'region', 'province',
        'locality', 'latitude', 'longitude', 'fl_furnished',
        'fl_floodzone', 'fl_double_glazing', 'fl_terrace',
        'fl_garden' ], axis=1, inplace=True
        )

'''
    ['price', 'property_type', 'zip_code', 'total_area_sqm', 'nbr_bedrooms',surface_land_sqm,
    'fl_open_fire', 'terrace_sqm', 'garden_sqm', 'fl_swimming_pool']
'''
###################### CLEAN UP DATASET - STOP ######################


###################### Feature Engineering - START ######################
# Step 3: Add column price_per_sqm to check for outliers
ds2['price_per_sqm'] = ds2['price']/ds2['total_area_sqm']

# check for outliers
ds2_stats = ds2['price_per_sqm'].describe()
print(ds2_stats)

# Step 3.1: min-Max of features ['total_area_sqm', 'nbr_bedrooms', 'terrace_sqm', 'garden_sqm', 'zip_code_encoded']
features_to_scale = ['total_area_sqm', 'nbr_bedrooms', 'terrace_sqm', 'garden_sqm', 'zip_code_encoded']

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the features to scale them between 0 and 1
cleaned_df[features_to_scale] = scaler.fit_transform(cleaned_df[features_to_scale])

# STEP 4: splitting data
#
# Filter rows where property_type is 'house'
houses_df = cleaned_df[cleaned_df['property_type'] == 'HOUSE'].drop(['property_type'], axis=1)

# Filter rows where property_type is 'apartment'
apartments_df = cleaned_df[cleaned_df['property_type'] == 'APARTMENT'].drop(['property_type'], axis=1)

# STEP 4.1: 80/20 dataset
X = houses_df.drop(['price'], axis=1)
y = houses_df['price']

# Split the training set into training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Step 5: Model performing
#
#
'''
model_SVR = svm.SVR()
model_SVR.fit(X_train,y_train)
y_pred = model_SVR.predict(X_test)

print('MAPE: ', mean_absolute_percentage_error(y_test, y_pred))

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)

'''


""" # Heatmap Seaborn
# Select only numerical features for correlation analysis
numerical_dataset = df.select_dtypes(include=['number'])

plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)

#plt.show() """

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Feature engineering example (log transformation for skewed features)
X['log_total_area_sqm'] = np.log1p(X['total_area_sqm'])
X['log_terrace_sqm'] = np.log1p(X['terrace_sqm'])
X['area_bedroom_interaction'] = X['total_area_sqm'] * X['nbr_bedrooms']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = GradientBoostingRegressor()

# Hyperparameter tuning with grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("Optimized MAE:", mae)
print("Optimized MAPE:", mape)

from sklearn.ensemble import RandomForestRegressor

model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, y_train)
Y_pred = model_RFR.predict(X_test)

mean_absolute_percentage_error(y_test, Y_pred)

from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error

model_SVR = svm.SVR()
model_SVR.fit(X_train,y_train)
Y_pred = model_SVR.predict(X_test)

print(mean_absolute_percentage_error(y_test, Y_pred))