import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cleaner
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
import joblib

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
ds1 = cleaner.clean_dataset(dataset, missing_threshold=0.3, retain_columns=['surface_land_sqm', 'latitude', 'longitude',])
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
        'locality', 'fl_furnished', 'zip_code',
        'fl_floodzone', 'fl_double_glazing', 'fl_terrace',
        'fl_garden' ], axis=1, inplace=True
        )
'''
    ['price', 'property_type', 'zip_code', 'total_area_sqm', 'nbr_bedrooms',surface_land_sqm,
    'fl_open_fire', 'terrace_sqm', 'garden_sqm', 'fl_swimming_pool']
'''
# Check if 'surface_land_sqm' is less than the sum of 'terrace_sqm' and 'garden_sqm'
# If so, replace the value with the sum of 'terrace_sqm' and 'garden_sqm'
ds2['surface_land_sqm'] = ds2.apply(
    lambda row: row['terrace_sqm'] + row['garden_sqm'] 
    if row['surface_land_sqm'] < row['terrace_sqm'] + row['garden_sqm'] 
    else row['surface_land_sqm'], 
    axis=1
)

# Drop 'terrace_sqm' and 'garden_sqm' columns
ds2 = ds2.drop(columns=['terrace_sqm', 'garden_sqm'])


# Split dataset
houses = ds2[ds2['property_type'] == 'HOUSE'].drop(columns=['property_type'])
apartements = ds2[ds2['property_type'] == 'APARTMENT'].drop(columns=['property_type'])
###################### CLEAN UP DATASET - STOP ######################



###################### Feature Engineering - START ######################

# Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations
# h_location_stats = houses['zip_code'].value_counts(ascending=False)
# a_location_stats = apartements['zip_code'].value_counts(ascending=False)
""" print(h_location_stats, a_location_stats)
# get the total of properties
print(h_location_stats.values.sum(), a_location_stats.sum()) # 39253 36253
# check which zip-code has more than 20 properties
print(len(h_location_stats[h_location_stats>10])) # 715 zip_codes
print(len(a_location_stats[a_location_stats>10])) # 410 zip_codes
print(len(h_location_stats + a_location_stats )) # 1073 total zip-codes
print(len(h_location_stats[h_location_stats<=10])) # 350
print(len(a_location_stats[a_location_stats<=10])) # 355 """

# Any location having less than 10 data points should be tagged as "9999" location. This way number of categories can be reduced by huge amount. 
# Later on when we do one hot encoding, it will help us with having fewer dummy columns
# h_location_stats_less_than_10 = h_location_stats[h_location_stats<=10]
# a_location_stats_less_than_10 = a_location_stats[a_location_stats<=10]

# houses['zip_code'] = houses['zip_code'].apply(lambda x: 9999 if x in h_location_stats_less_than_10 else x)
# apartements['zip_code'] = apartements['zip_code'].apply(lambda x: 9999 if x in a_location_stats_less_than_10 else x)

# We can assume that the minimum square m per bedroom is 30. A small studio must be atleast 25m2 to have a bedroom inside
# If you have for example 80 m2 apartment with 4 bedrooms than that seems suspicious and can be removed as an outlier. 
# We will remove such outliers by keeping our minimum thresold per bhk to be 25m2
houses = houses[~(houses.total_area_sqm/houses.nbr_bedrooms<25)]
apartements = apartements[~(apartements.total_area_sqm/apartements.nbr_bedrooms<25)]

print(houses.head(10))

# Step 3.1: min-Max of features ['total_area_sqm', 'nbr_bedrooms', 'terrace_sqm', 'garden_sqm', 'zip_code_encoded']
# features_to_scale = ['total_area_sqm', 'nbr_bedrooms', 'terrace_sqm', 'garden_sqm']

# STEP 4: splitting data

# STEP 4.1: 80/20 dataset
Xh = houses.drop(['price'], axis=1)
yh = houses['price']
Xa = apartements.drop(['price'], axis=1)
ya = apartements['price']

# Split the training set into training and validation set for house and apartment
Xh_train, Xh_test, yh_train, yh_test = train_test_split(Xh, yh, test_size=0.2, random_state=0)
Xa_train, Xa_test, ya_train, ya_test = train_test_split(Xa, ya, test_size=0.2, random_state=0)

# Best parameters from previous GridSearchCV run
best_params = {
    'colsample_bytree': 1.0,
    'gamma': 0,
    'learning_rate': 0.1, #lowered from 0.2
    'max_depth': 5,
    'n_estimators': 500, # 300
    'reg_alpha': 0.1, #0
    'reg_lambda': 1.5, #1
    'subsample': 0.6
}

# Initialize the XGBoost model with best parameters and GPU configuration
model_houses = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    tree_method='hist',  # Enables GPU usage
    device="cuda",
    **best_params
)

model_apartments = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    tree_method='hist',
    **best_params
)

# Train the model
model_houses.fit(Xh_train, yh_train)
model_apartments.fit(Xa_train, ya_train)

# Evaluate the model
print("XGBoost Training Score:", model_houses.score(Xh_train, yh_train))
print("XGBoost Test Score:", model_houses.score(Xh_test, yh_test))

# Cross-validation for further evaluation
xgb_cross_val_scores = cross_val_score(model_houses, Xh, yh, cv=10)
print("Mean Cross-Validation Score for XGBoost:", xgb_cross_val_scores.mean())

# Make predictions using the trained model
y_pred = model_houses.predict(Xh_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(yh_test, y_pred)
print("Mean Absolute Error for test (MAE):", mae)

rmse = root_mean_squared_error(yh_test, y_pred)
print("Root Mean Squared Error for test (RMSE):", rmse)

# Save the trained model to a file
joblib.dump(model_houses, 'model/xgboost_model_houses.pkl')
print("Model saved as 'xgboost_model_houses.pkl'")

# Save the trained model for apartments
joblib.dump(model_apartments, 'model/xgboost_model_apartments.pkl')
print("Model for apartments saved as 'xgboost_model_apartments.pkl'")