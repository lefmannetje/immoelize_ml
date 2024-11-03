import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ds = pd.read_csv("data/cleaned_dataset.csv")
ds['price_sqm'] = ds['price']/ds['total_area_sqm']

zipcode_stats = ds.groupby('zip_code')['zip_code'].agg('count').sort_values(ascending=False)
zip_code_less_than_20 = zipcode_stats[zipcode_stats<=20]

ds.zip_code = ds.zip_code.apply(lambda x: '9999' if x in zip_code_less_than_20 else x)

ds = ds[~(ds.total_area_sqm/ds.nbr_bedrooms<30)]

def remove_pps_outliers(ds):
    df_out = pd.DataFrame()
    for key, subdf in ds.groupby('zip_code'):
        m = np.mean(subdf.price_sqm)
        st = np.std(subdf.price_sqm)
        reduced_df = subdf[(subdf.price_sqm>(m-st)) & (subdf.price_sqm<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df = remove_pps_outliers(ds)

df = df.drop("price_sqm", axis=1)

df['zip_code'] = df['zip_code'].astype(str)
dummies = pd.get_dummies(df.zip_code)
# Convert True/False to 1/0
dummies = dummies.astype(int)

print(df.columns)