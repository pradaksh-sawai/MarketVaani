from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime
import yfinance as yf

# Initialize Flask app
app = Flask(__name__)

# Load your trained model (ensure the model is saved as 'saved_model.h5')
model = load_model('model/saved_model.h5')

# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Function to preprocess data (scaling)
def preprocess_data(input_data):
    input_data = scaler.fit_transform(np.array(input_data).reshape(-1, 1))
    return input_data

# Function to make predictions
def make_prediction(data):
    # Assuming data is scaled and reshaped correctly for the LSTM model
    data = np.reshape(data, (1, data.shape[0], 1))
    prediction = model.predict(data)
    return scaler.inverse_transform(prediction)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock = request.form['stock']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    try:
        # Fetch historical stock data using yfinance
        df = yf.download(stock, start=start_date, end=end_date)
        
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No data found for the given stock and date range.'})
        
        # Use only the 'Close' prices for prediction
        input_data = df['Close'].values[-5:]  # Get the last 5 closing prices for prediction
        processed_data = preprocess_data(input_data)
        
        # Make prediction
        prediction = make_prediction(processed_data)
        
        return jsonify({'status': 'success', 'prediction': str(prediction[0][0])})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
