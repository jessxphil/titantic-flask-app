# Titantic Prediction App Deployed in Herouku 
Create a Flask API to Deploy Titanic Prediction Model Using Heroku

# Deploy ML model with Flask to Heroku

Click the button below to quickly clone and deploy into your own Heroku acount.
If you don't have one it'll prompt you to setup a free one.

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

Once deployed to your Heroku instance run the following:

```bash
curl -s -XPOST 'https://<name-of-your-heroku-app>.herokuapp.com/' -d '{"Pclass":3,"Age":2,"SibSp":1,"Fare":50}' -H 'accept-content: application/json'
```

Alternatively a simple python script:

```python
import requests
import json
url = 'https://<name-of-your-heroku-app>.herokuapp.com/'
data = {"Pclass":3, "Age":2, "SibSp":1, "Fare":50}
response = requests.post(url, json.dumps(data))
print(response.json())
```


