import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Load data and trained model (example assumes you've saved them)
@st.cache_data
def load_data(filepath):
    """
    Load the household power consumption dataset
    """
    columns = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power',
               'Voltage', 'Global_intensity', 'Sub_metering_1',
               'Sub_metering_2', 'Sub_metering_3']

    # Read the data with semi-colon separator and specified column names
    df = pd.read_csv(filepath, sep=';', names=columns, na_values=['?'])

    # Parse datetime explicitly
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce', dayfirst=True)

    # Drop rows where datetime conversion failed
    df = df.dropna(subset=['datetime'])

    # Convert power consumption columns to numeric, setting errors='coerce' to handle non-numeric entries
    cols_to_numeric = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any remaining rows with NaN values after conversion
    df = df.dropna()

    # Extract time-based features
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['day_of_week'] = df['datetime'].dt.dayofweek

    return df

# Load model
@st.cache_resource
def load_model():
    # Example of loading a pre-trained model if saved
    model = RandomForestRegressor()  # Load the actual saved model
    return model

# Prediction function
def predict_energy_consumption(model, data):
    return model.predict(data)

# Streamlit App
st.title("Energy Consumption Dashboard")

# Sidebar for uploading data
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    data = load_data(uploaded_file)
    st.write("Data Preview:", data.head())

    # Allow users to visualize historical data
    st.subheader("Historical Data Visualization")
    fig1 = px.line(data, x='datetime', y='Global_active_power', title="Global Active Power Over Time")
    st.plotly_chart(fig1)

    # Prepare feature columns for prediction
    feature_columns = ['hour', 'day', 'month', 'year', 'day_of_week', 'Global_reactive_power', 'Voltage',
                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    data_features = data[feature_columns]

    # Load the pre-trained model and make predictions
    model = load_model()
    predictions = predict_energy_consumption(model, data_features)

    # Add predictions to data
    data['Predicted_Global_active_power'] = predictions

    # Visualization of Predictions vs. Actuals
    st.subheader("Predicted vs Actual Energy Consumption")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data['datetime'], y=data['Global_active_power'], mode='lines', name='Actual'))
    fig2.add_trace(go.Scatter(x=data['datetime'], y=data['Predicted_Global_active_power'], mode='lines', name='Predicted'))
    st.plotly_chart(fig2)

    # Performance Metrics
    st.subheader("Model Performance Metrics")
    rmse = np.sqrt(mean_squared_error(data['Global_active_power'], data['Predicted_Global_active_power']))
    mae = mean_absolute_error(data['Global_active_power'], data['Predicted_Global_active_power'])
    r2 = r2_score(data['Global_active_power'], data['Predicted_Global_active_power'])

    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**R2 Score:** {r2:.2f}")
