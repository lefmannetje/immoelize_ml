import joblib
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from geopy.geocoders import Nominatim

# Function to get coordinates from postal code using geopy
def get_coordinates(postal_code):
    geolocator = Nominatim(user_agent="property_predictor")
    location = geolocator.geocode(f"{postal_code}, Belgium")
    if location:
        return location.latitude, location.longitude
    else:
        print("Postal code not found.")
        return None, None

# Load models
model_houses = joblib.load('model/xgboost_model_houses.pkl')
model_apartments = joblib.load('model/xgboost_model_apartments.pkl')

# Create a simple tkinter form
root = tk.Tk()
root.withdraw()  # Hide the root window

property_type = simpledialog.askstring("Input", "Enter property type (HOUSE/APARTMENT):")

if property_type and property_type.strip().upper() in ['HOUSE', 'APARTMENT']:
    total_area = simpledialog.askfloat("Input", "Enter total area (sqm):")
    nbr_bedrooms = simpledialog.askinteger("Input", "Enter number of bedrooms:")
    surface_land = simpledialog.askfloat("Input", "Enter surface land (sqm):")
    fl_open_fire = simpledialog.askinteger("Input", "Has open fire (1 for Yes, 0 for No):")
    fl_swimming_pool = simpledialog.askinteger("Input", "Has swimming pool (1 for Yes, 0 for No):")
    postal_code = simpledialog.askinteger("Input", "Enter the Belgium postal code (e.g., 9000 for Gent):")

    # Get latitude and longitude from postal code
    latitude, longitude = get_coordinates(postal_code)

    if latitude is not None and longitude is not None:
        # Prepare the feature array
        input_features = np.array([[total_area, nbr_bedrooms, surface_land, fl_open_fire, fl_swimming_pool, latitude, longitude]])

        if property_type.strip().upper() == 'HOUSE':
            predicted_price = model_houses.predict(input_features)
            print("Predicted price for the house:", predicted_price[0])
        else:
            predicted_price = model_apartments.predict(input_features)
            print("Predicted price for the apartment:", predicted_price[0])
    else:
        print("Unable to retrieve latitude and longitude for the given postal code.")
else:
    print("Invalid property type entered.")
