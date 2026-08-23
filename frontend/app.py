import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Sales Revenue Prediction")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for property features
Product_Weight = st.number_input("Product_Weight", min_value=0, value=1)
Product_Sugar_Content = st.selectbox("Product_Sugar_Content", ["low sugar", "regular", "no sugar"])
Product_Allocated_Area = st.number_input("Product_Allocated_Area", min_value=1, value=.01)
Product_MRP = st.number_input("Product_MRP", min_value=0, value=1)
Store_Size = st.selectbox("Store_Size", ["High", "Medium","Low"])
Store_Location_City_Type = st.selectbox("Store_Location_City_Type", ["Tier 1", "Tier 2","Tier 3"])
Store_Type = st.selectbox("Store_Type",["Departmental Store", "Supermarket Type 1", "Supermarket Type 2", "Food Mart"])
Product_Id_char = st.number_input("Product_Id_char")
Store_Age_Years = st.number_input("Store_Age_Years", min_value=0, step=1, value=1)
Product_Type_Category = st.selectbox("Product_Type_Category",["meat", "snack foods", "hard drinks", "dairy", "canned", "soft drinks", "health and hygiene", "baking goods", "bread", "breakfast", "frozen foods", "fruits and vegetables", "household", "seafood", "starchy foods", "others"])

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    'Product_Weight': Product_Weight, #
    'Product_Sugar_Content': Product_Sugar_Content, # low sugar, regular, and no sugar
    'Product_Allocated_Area': Product_Allocated_Area, # Broad category for each product like meat, snack foods, hard drinks, dairy, canned, soft drinks, health and hygiene, baking goods, bread, breakfast, frozen foods, fruits and vegetables, household, seafood, starchy foods, others
    'Product_MRP': Product_MRP,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,  # Convert to 't' or 'f' # Type of city in which the store is located, like Tier 1, Tier 2, and Tier 3
    'Store_Type': Store_Type, # Type of store depending on the products that are being sold there, like Departmental Store, Supermarket Type 1, Supermarket Type 2, and Food Mart
    'Product_Id_char': Product_Id_char,
    'Store_Age_Years': Store_Age_Years,
    'Product_Type_Category': Product_Type_Category
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/sales", json=input_data.to_dict(orient='records')[0])  # Send data to Flask API
    if response.status_code == 200:
        prediction = response.json()['Predicted Price (in dollars)']
        st.success(f"Predicted Sales Revenue (in dollars): {prediction}")
    else:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/salesbatch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")
