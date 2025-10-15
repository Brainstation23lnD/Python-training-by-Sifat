### 📘 **README.md**

# 🍕 Restaurant Review & Weather Assistant

An **AI-powered interactive assistant** that can:
- Answer questions about restaurants (food quality, service, pricing, etc.)
- Retrieve and summarize real customer reviews using **RAG (Retrieval-Augmented Generation)**
- Provide **live weather information** for any city via an integrated API
- Combine both tools through a **LangChain ReAct agent** for reasoning and decision-making

---

## 🚀 Features

- **LangChain ReAct Agent**: Combines reasoning with action to decide when to use tools.
- **RAG Pipeline**: Fetches relevant restaurant reviews from a vector database.
- **Weather Tool**: Retrieves real-time weather data via API.
- **Local LLM Integration**: Runs on `Ollama (llama3.2)` for on-device inference.
- **Interactive CLI Mode**: Chat directly with the assistant in your terminal.

---

## 🧠 Architecture

```
User Input
│
▼
[ LangChain ReAct Agent ]
├── Tool 1 → Weather API (get_weather_info)
├── Tool 2 → Restaurant RAG (ask_restaurant)
▼
LLM (Ollama Llama3.2)
▼
Final Answer → Console Output

````

---

## 🧩 Components Overview

| File                                 | Description                                      |
|--------------------------------------|--------------------------------------------------|
| **main.py**                          | Core application & agent loop                    |
| **vector.py**                        | Loads vector store and defines retriever for RAG |
| **weather_api.py**                   | Fetches live weather data                        |
| **realistic_restaurant_reviews.csv** | Dataset used for restaurant review retrieval     |
| **requirements.txt**                 | Python dependencies                              |
| **README.md**                        | Project documentation                            |

---

## ⚙️ Installation

### Cone the Repository
```bash
git clone https://github.com/GenesisBlock3301/restuarant_review_management.git
cd restuarant_review_management
````

### Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # For Linux/Mac
venv\Scripts\activate        # For Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Ollama Locally

Ensure you have [Ollama](https://ollama.ai/download) installed and running:

```bash
ollama run llama3.2
```

---

## ▶️ Run the Application

Simply run:
```bash
python main.py
```
---
## 💬 Example Interactions

```
🍕 Pizza Restaurant & Weather Assistant
Ask me about restaurants or weather information!
Type 'q' or 'quit' to exit.

Ask your question: What do people say about the pizza quality?

🤔 Processing: What do people say about the pizza quality?
✅ Final Answer:
Customers love the crispy crust and fresh toppings. Several reviews mention fast delivery and great taste.
```

---

### 🌦️ Weather Example

```
Ask your question: What's the weather like in Dhaka today?

🤔 Processing: What's the weather like in Dhaka today?
✅ Final Answer:
Currently 31°C in Dhaka with light rain and 80% humidity.
```

---

## 🧱 Project Structure

```
📂 restuarant_review_management
 ┣ 📜 main.py
 ┣ 📜 vector.py
 ┣ 📜 weather_api.py
 ┣ 📜 realistic_restaurant_reviews.csv
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

---

## 🧰 Tools & Technologies

* **Python 3.10+**
* **LangChain**
* **Ollama Llama 3.2**
* **Vector Store**
* **Weather API**
* **ReAct Agent Framework**

---

## 👨‍💻 Author

**GenesisBlock3301**
🌐 [GitHub Profile](https://github.com/GenesisBlock3301)

