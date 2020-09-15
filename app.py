# IMPORTS
import pandas as pd
from flask import Flask, jsonify, request
import pickle
import flask_monitoringdashboard as dashboard

# LOAD MODEL
model = pickle.load(open('model.pkl','rb'))

# CREATE API ROUTE
app = Flask(__name__)

# API ROUTE
@app.route('/', methods=['POST'])

# PREDICTION MODEL
def predict():

    # We can read the content of the server’s response. 
    # Response Content¶
    # get data
    data = request.get_json() # Converts incoming request data from JSON object into Python data   
    
    # request.get_json() converts the JSON object into Python data for us. JSON decoder
    # Let's assign the incoming request data to variables and return them 
    # by making the following changes to our json-example route.

    # convert json data from dict into dataframe
    data.update((x, [y]) for x, y in data.items()) # Insert an item to the dict using .update()
    data_df = pd.DataFrame.from_dict(data)         # Convert dict to dataframe

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)



if __name__ == '__main__':
    app.run(port = 5000, debug=True)

