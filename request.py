import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'title':'blabla', 'body':'bla'})

print(r.json())