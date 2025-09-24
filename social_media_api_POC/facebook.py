import requests
import json

page_id = "845728675279751"
access_token = "EAAMcUsxmh68BPlGZAVw4OYIyUBpiwZBOmA1Bywb7K9qtrquNZBXZAOx7mUTt5kvwq6XBN7DWdttQwnfUGxkBohjmUPIy8q9T68CQtlu5eoLGTP73aPpsXIcS4lhZBuoORBKt30AGCVt6fZCxRzYhcAHKTGJbgscbZBHohEqxNkrlwnr3QBeomcuNCBrkObYr4y7CTDv93d5GXciXWlHnjLLjeD7CV4p5PJxmDIGSiPb3rYZD"
url = f"https://graph.facebook.com/v18.0/{page_id}"
params = {
    "fields": "name,about,category,fan_count,posts",
    "access_token": access_token
}


response = requests.get(url, params=params)
data = response.json()

print(json.dumps(data, indent=4, ensure_ascii=False))
