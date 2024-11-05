# Real Estate Price Prediction for Immo Eliza

![Feature Importances](data/immo-search.webp)  

##  Context

Immo Eliza wanted a prediction tool to assist both buyers and sellers in Belgium to make market predictions of their real astate. Based on 6 features it will make the prediction
### Features
- area surface
- land surface
- number of bedrooms
- has open fire
- has swimming pool
- location


## Data

- **Dataset**: We used `data/properties.csv` as datset. After the cleanup we held on to +/- 60k property listings in Belgium, houses & apartments. The `data/cleaned_Dataset.csv` is alrdy cleaned and has the best score.

## Model Details

- **Tested Models**: Linear Regression, Random Forest, XGBRegressor.
- **Chosen Model**: XGBRegressor was selected for its best score.

## Performance

The XGBRegressor model achieved 
- XGBoost Training Score: 0.9316877515763166
- XGBoost Test Score: 0.8129196050078894
- Mean Cross-Validation Score for XGBoost: 0.78153067585399
- Mean Absolute Error for test (MAE): 99075.4958276164
- Root Mean Squared Error for test (RMSE): 180832.77556397254

## Limitations

- **Data Quality**: Model accuracy relies heavily on data quality. Unique properties may be less accurately predicted.
- **Time-Sensitive Factors**: Does not consider market trends or economic conditions.
- **Region Specific**: Trained on Belgian data, which may not generalize to other regions.

## Future Work

- Explore more advanced models and feature engineering techniques.
- getting an Up-to-date dataset and clean it better. 

## Usage Guide

### Dependencies

Install dependencies from `requirements.txt`. Main libraries: `pandas`, `scikit-learn`, `joblib`, `xgboost`.

### Cleaning and training the Model

Run `main.py` to train the model. You can un-comment code if you want to experiment with better datacleaning. It will improve score, but it will also remove a lot of data.

### Generating Predictions

Use `predict.py`. It will popup a form where you can fill in the correct features to get a prediciton in temminal screen
