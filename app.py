import pickle
from flask import Flask, request, app, jsonify, render_template
import numpy as np

app = Flask(__name__)                               # Initializing the flask app
model = pickle.load(open('model.pkl', 'rb'))        # Loading the model

@app.route('/')                                     # Defining the home page of our web-app
def home():                                         # Defining the function to be called when home is accessed
    return render_template('home.html')             # Rendering the home.html file

@app.route('/prediction_api', methods = ['POST'])   # Defining the predict_api page of our web-app
def predict_api():                                  # Defining the function to be called when predict_api is accessed
    data = request.json['data']                     # Getting the data from the POST request.
    new_data = np.array(data.values()).reshape(1,-1)# Converting the data into a numpy array for processing.
    output = model.predict(new_data)                # Getting the model's prediction on the data.
    # print(output[0])
    return jsonify(output[0])                      # Returning the output as a json file.
    # return render_template('home.html', f'The predicted value is {output[0]}.')

if __name__ == "__main__":                         # Defining the main function
    app.run(debug = True)                          # Running the app in debug mode