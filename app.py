# import modules
import pandas as pd
from flask import Flask, request, jsonify
import pickle
from pickle import load
import os, sys
import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.models import model_from_yaml
import json

# LOAD MODELS
# yaml_file = open('model_files/model.yaml', 'rb')  # load YAML and create model         
# loaded_model_yaml = yaml_file.read() 
# yaml_file.close()
# loaded_model = model_from_yaml(loaded_model_yaml) 
# print("Loaded model from disk w/ YAML")

# loaded_model.load_weights('model_files/model.h5') # load weights into new model       
# print("Loaded model from disk w/ HdF5")

# json_file = open('model_files/model.json', 'rb')  # load json and create model
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# print("Loaded model from disk w/ JSON")

# LOAD MODEL
model = pickle.load(open('model.pkl','rb')) # TT

# CREATE API ROUTE
app = Flask(__name__)

# API ROUTE
@app.route('/', methods=['POST'])

# PREDICTION MODEL
def predict():

    # Converts incoming request data from JSON object into Python data
    # data = request.get_json(force=False)  # Ignore the mimetype and always try to parse data as JSON.
    data = request.get_json() ## CC = Removed (force = False)
    print(type(data), data)

    data = json.loads(data)          # load str into dict to use it

   # convert json data from dict into dataframe
    data.update((x, [y]) for x, y in data.items()) # Insert an item to the dict using .update()
    data_df = pd.DataFrame.from_dict(data)         # Convert dict to dataframe

    # data_arr = np.array(data_df)                          # Convert data to array
    # print("Convert sample to array", data_arr)

    # Transform Data from 2D to 3D for CNN 
    # data_arr = data_arr.reshape(data_arr.shape[0], data_arr.shape[1], 1)

    # Predict using model
    # yTween = loaded_model.predict(data_arr)
    yTween = model.predict(data_df)
    print("Tween prediced values: unscaled , complete.")

    #the curly brackets turn object into dict
    output = {'results': int(yTween[0])} # Added:  output = {'results': int(result[0])}
    print(output, type(output))

    return jsonify(results=output)

if __name__ == '__main__': 
    app.run(port = 5000, debug=True) # Added: app.run(port = 5000, debug=True)



