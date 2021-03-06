# IMPORTS
import pandas as pd
from flask import Flask, jsonify, request
import pickle

# LOAD MODEL
model = pickle.load(open('model.pkl','rb'))

# APP
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)
        
    # data = json.loads(data) # Code Works? No

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)
    
    # predictions
    result = model.predict(data_df)

    # send back to browser
    # output = {'results': int(result[0])}
    # output = {'results': int(result)} # Code Works? Yes
    
    # Convert array to Json 
    output = result.tolist()
    output = json.dumps(result)
    output = {'results': (output)}
    
    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
