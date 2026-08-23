# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
sales_revenue_predictor_api = Flask("SuprKart Sales Revenue Predictor")

# Load the trained machine learning model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page (GET request)
@sales_revenue_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Revenue Prediction API!"

# Define an endpoint for single store sales revenue prediction (POST request)
@sales_revenue_predictor_api.post('/v1/sales')
def predict_rental_price():
    """
    This function handles POST requests to the '/v1/sales' endpoint.
    It expects a JSON payload containing property details and returns
    the predicted rental price as a JSON response.
    """
    # Get the JSON data from the request body
    property_data = request.get_json()

    # Extract relevant features from the JSON data
    payload = {
      "Product_Weight": 12.66,
      "Product_Sugar_Content": "Low Sugar",
      "Product_Allocated_Area": 0.027,
      "Product_MRP": 117.08,
      "Store_Size": "Medium",
      "Store_Location_City_Type": "Tier 2",
      "Store_Type": "Supermarket Type2",
      "Product_Id_char": "FD",
      "Store_Age_Years": 16,
      "Product_Type_Category": "Non Perishables"
   }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([payload])

    # Make prediction (get sales_revenue)
    predicted_sales_revenue = model.predict(input_data)[0]

    # Calculate actual revenue
    predicted_revenue = np.exp(predicted_sales_revenue)

    # Convert predicted_revenue to Python float
    predicted_revenue = round(float(predicted_revenue), 2)
    # The conversion above is needed as we convert the model prediction (sales revenue) to actual revenue using np.exp, which returns predictions as NumPy float32 values.
    # When we send this value directly within a JSON response, Flask's jsonify function encounters a datatype error

    # Return the actual price
    return jsonify({'Predicted Price (in dollars)': predicted_revenue})


# Define an endpoint for batch prediction (POST request)
@sales_revenue_predictor_api.post('/v1/salesbatch')
def predict_sales_revenue_batch():
    """
    This function handles POST requests to the '/v1/salesbatch' endpoint.
    It expects a CSV file containing property details for multiple properties
    and returns the predicted sales revenues as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all properties in the DataFrame (get log_prices)
    predicted_sales_revenues = model.predict(input_data).tolist()

    # Calculate actual prices
    predicted_revenues = [round(float(np.exp(sales_revenue)), 2) for sales_revenue in predicted_sales_revenues]

    # Create a dictionary of predictions with store IDs as keys
    store_ids = input_data['Store_Id'].tolist()  # Assuming 'id' is the property ID column
    output_dict = dict(zip(store_ids, predicted_revenues))  # Use actual prices

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    sales_revenue_predictor_api.run(debug=True)
