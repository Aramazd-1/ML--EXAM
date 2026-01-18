import requests

url = "https://www.fema.gov/api/open/v2/FimaNfipClaims?$top=1"
js = requests.get(url, timeout=60).json()
one = js["FimaNfipClaims"][0]
print(sorted(one.keys()))