from flask import Flask, render_template, request
import joblib
import numpy as np
from geopy.geocoders import Nominatim

# Initialize Flask app
app = Flask(__name__)

# Load the models
model_houses = joblib.load('model/xgboost_model_houses.pkl')
model_apartments = joblib.load('model/xgboost_model_apartments.pkl')

# Function to get coordinates from postal code using geopy
def get_coordinates(postal_code):
    geolocator = Nominatim(user_agent="property_predictor")
    location = geolocator.geocode(f"{postal_code}, Belgium")
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

@app.route('/', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # Get form data
        property_type = request.form.get('property_type')
        total_area = float(request.form.get('total_area'))
        nbr_bedrooms = int(request.form.get('nbr_bedrooms'))
        surface_land = float(request.form.get('surface_land'))
        fl_open_fire = int(request.form.get('fl_open_fire'))
        fl_swimming_pool = int(request.form.get('fl_swimming_pool'))
        postal_code = request.form.get('postal_code')

        # Get latitude and longitude from postal code
        latitude, longitude = get_coordinates(postal_code)
        
        if latitude is not None and longitude is not None:
            # Prepare the feature array
            input_features = np.array([[total_area, nbr_bedrooms, surface_land, fl_open_fire, fl_swimming_pool, latitude, longitude]])

            # Choose the model based on property type
            if property_type == 'HOUSE':
                predicted_price = model_houses.predict(input_features)
            else:
                predicted_price = model_apartments.predict(input_features)
            
            # Return the result to the user
            return render_template('form.html', prediction=f"Predicted price: €{predicted_price[0]:,.2f}")
        else:
            return render_template('form.html', error="Unable to retrieve location. Please check the postal code.")

    # Render the form initially
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
