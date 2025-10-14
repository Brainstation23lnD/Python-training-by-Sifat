import requests


def get_weather(city: str) -> str:
    """Fetch current weather from a public API (example: OpenWeatherMap)"""
    city = city.strip("'\'")
    API_KEY = "0d10c44bb9633604f35d97028b66c3fb"  # OpenWeatherMap API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("main"):
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"The current temperature in {city} is {temp}°C with {desc}."
        else:
            return f"Could not fetch weather for {city}."
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    print(get_weather("dhaka"))