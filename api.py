from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf

# Load the pre,-trained CNN model
modelCnn = tf.keras.models.load_model('modelCnn.h5')
modelCnnLstm = tf.keras.models.load_model('modelCnnLstm.h5')

# Create a Flask application
app = Flask(__name__)


# Define a route for the API endpoint ClassifyCnn
@app.route('/classifyCnn', methods=['POST'])
def classifyCnn():
    # Get the ECG signal data from the request
    signal_data = request.json['signal_data']

    # Preprocess the signal data
    signal_data = np.array(signal_data)
    signal_data = signal_data.reshape((1, signal_data.shape[0], 1))

    # Make a prediction using the pre,-trained model
    prediction = modelCnn.predict(signal_data)
    prediction = prediction.round()
    # print(prediction)
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})


# Define a route for the API endpoint  ClassifyCnnLstm
@app.route('/classifyCnnLstm', methods=['POST'])
def classifyCnnLstm():
    # Get the ECG signal data from the request
    signal_data = request.json['signal_data']

    # Preprocess the signal data
    signal_data = np.array(signal_data)
    signal_data = signal_data.reshape((1, signal_data.shape[0], 540, 1))

    # Make a prediction using the pre,-trained model
    prediction = modelCnnLstm.predict(signal_data)
    prediction = prediction.round()
    # print(prediction)
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
