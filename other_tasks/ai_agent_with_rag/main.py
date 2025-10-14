from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from vector import retriever
from weather_api import get_weather

# Initialize the model
model = OllamaLLM(model="llama3.2")

# RAG chain: prompt + LLM
template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


def ask_restaurant(query: str) -> str:
    """Answer questions about pizza restaurants using reviews."""
    try:
        # Retrieve top 5 relevant reviews
        reviews = retriever.invoke(query)
        # Format the reviews for better readability
        if isinstance(reviews, list):
            formatted_reviews = "\n".join([f"- {review}" for review in reviews])
        else:
            formatted_reviews = str(reviews)

        response = model.invoke(prompt.format(reviews=formatted_reviews, question=query))
        return response
    except Exception as e:
        return f"Error retrieving restaurant information: {str(e)}"


def get_weather_info(city: str) -> str:
    """Get weather information for a city."""
    try:
        return get_weather(city)
    except Exception as e:
        return f"Error retrieving weather information: {str(e)}"


# Define tools
tools = [
    Tool(
        name="weather_api",
        func=get_weather_info,
        description="Get current weather information for a specific city. Input should be the city name as a string."
    ),
    Tool(
        name="restaurant_rag",
        func=ask_restaurant,
        description="Answer questions about pizza restaurants using customer reviews and restaurant information. Use this for questions about food quality, service, menu items, pricing, or general restaurant experience."
    )
]

# Create REACT prompt template
react_prompt = PromptTemplate.from_template("""
Answer the question using the following tools if needed:

{tools}

Follow this format:

Thought: Explain which tool to use or if no tool is needed
Action: Name of the tool to use [{tool_names}] or "None"
Action Input: Input for the tool (plain text, no quotes) or "None"
Observation: Result of the action or "None"

If you have the answer, stop and output:
Thought: I now know the final answer
Final Answer: [your clear, concise answer]

Do NOT continue after a Final Answer. Do NOT process unrelated questions.

Question: {input}
Thought: {agent_scratchpad}
""")

# Create the agent
agent = create_react_agent(model, tools, react_prompt)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,  # Reduce to minimize unnecessary steps
    handle_parsing_errors=True,
    early_stopping_method="force",
    return_intermediate_steps=True
)

def test_weather_api():
    """Test function to debug weather API"""
    print("🧪 Testing Weather API...")
    print("=" * 40)

    test_cities = ["Dhaka", "dhaka", "Dhaka,BD", "London"]

    for city in test_cities:
        print(f"\n🔍 Testing: '{city}'")
        result = get_weather(city)
        print(f"📋 Result: {result}")
        print("-" * 30)


def main():
    """Main function to run the interactive agent."""

    # Uncomment the line below to test weather API first
    # test_weather_api()
    # return

    print("🍕 Pizza Restaurant & Weather Assistant")
    print("Ask me about restaurants or weather information!")
    print("Type 'q' or 'quit' to exit.\n")
    print("💡 Tip: Try 'test weather' to debug weather API")

    while True:
        print("\n" + "=" * 50)
        try:
            question = input("Ask your question: ").strip()

            if question.lower() in ['q', 'quit', 'exit']:
                print("Goodbye! 👋")
                break

            # Special debug command
            if question.lower() == 'test weather':
                test_weather_api()
                continue

            if not question:
                print("Please enter a question.")
                continue

            print(f"\n🤔 Processing: {question}")
            print("-" * 30)

            # Invoke the agent with better error handling
            try:
                response = agent_executor.invoke({"input": question})

                # Check if we have a proper output
                if response.get('output'):
                    print(f"\n✅ Final Answer:\n{response['output']}")
                else:
                    # If no output, try to get info from intermediate steps
                    if response.get('intermediate_steps'):
                        print(f"\n⚠️ Agent completed but without final answer. Last tool result:")
                        last_step = response['intermediate_steps'][-1]
                        if len(last_step) > 1:
                            print(last_step[1])  # Tool output
                    else:
                        print("\n❌ Agent stopped without providing an answer.")

            except Exception as e:
                if "Agent stopped due to iteration limit" in str(e):
                    print(f"\n⚠️ Agent took too many steps. Let me try a simpler approach...")
                    # Fallback: try to call tools directly based on question content
                    question_lower = question.lower()
                    if any(word in question_lower for word in ['weather', 'temperature', 'climate']):
                        # Extract city name from question
                        cities = ['dhaka', 'london', 'new york', 'tokyo', 'paris', 'sydney']
                        city_found = None
                        for city in cities:
                            if city in question_lower:
                                city_found = city.title()
                                break

                        if city_found:
                            print(f"🌤️ Getting weather for {city_found}...")
                            result = get_weather_info(city_found)
                            print(f"\n✅ Weather Info:\n{result}")
                        else:
                            print("Could not identify city from question. Please specify a city name.")

                    elif any(word in question_lower for word in ['restaurant', 'pizza', 'food', 'menu']):
                        print("🍕 Searching restaurant information...")
                        result = ask_restaurant(question)
                        print(f"\n✅ Restaurant Info:\n{result}")

                    else:
                        print(f"❌ Error: {str(e)}")
                else:
                    print(f"\n❌ Error: {str(e)}")
                    print("Please try again with a different question.")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    main()
