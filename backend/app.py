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
      "Product_Weight": property_data['Product_Weight'],
      "Product_Sugar_Content": property_data['Product_Sugar_Content'],
      "Product_Allocated_Area": property_data['Product_Allocated_Area'],
      "Product_MRP": property_data['Product_MRP'],
      "Store_Size": property_data['Store_Size'],
      "Store_Location_City_Type": property_data['Store_Location_City_Type'],
      "Store_Type": property_data['Store_Type']
   }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([payload])

    # Make prediction (get sales_revenue)
    predicted_sales_revenue = model.predict(input_data)[0]

    # Convert NumPy value to Python float
    predicted_revenue = round(float(predicted_sales_revenue), 2)

    # Return prediction
    return jsonify({
        'Predicted Sales Revenue': predicted_revenue
    })


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
    predicted_revenues = [round(float(sales_revenue), 2) for sales_revenue in predicted_sales_revenues]

    output_dict = {
      "predictions": predicted_revenues
    }

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    sales_revenue_predictor_api.run(debug=True)