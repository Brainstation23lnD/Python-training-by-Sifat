from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from vector import retriever
from weather_api import get_weather

# -------------------------
# Initialize LLM
# -------------------------
model = OllamaLLM(model="llama3.2")

# -------------------------
# RAG Chain Prompt
# -------------------------
template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# -------------------------
# Helper Functions
# -------------------------
def ask_restaurant(query: str) -> str:
    """Answer questions about pizza restaurants using reviews."""
    try:
        reviews = retriever.invoke(query)
        formatted_reviews = (
            "\n".join([f"- {review}" for review in reviews])
            if isinstance(reviews, list)
            else str(reviews)
        )
        return model.invoke(prompt.format(reviews=formatted_reviews, question=query))
    except Exception as e:
        return f"Error retrieving restaurant information: {str(e)}"


def get_weather_info(city: str) -> str:
    """Get weather information for a city."""
    try:
        return get_weather(city)
    except Exception as e:
        return f"Error retrieving weather information: {str(e)}"


# -------------------------
# Tools Definition
# -------------------------
tools = [
    Tool(
        name="weather_api",
        func=get_weather_info,
        description="Get current weather information for a specific city."
    ),
    Tool(
        name="restaurant_rag",
        func=ask_restaurant,
        description="Answer questions about pizza restaurants using customer reviews."
    )
]

# -------------------------
# REACT Prompt Template
# -------------------------
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

# -------------------------
# Create Agent & Executor
# -------------------------
agent = create_react_agent(model, tools, react_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True,
    early_stopping_method="force",
    return_intermediate_steps=True
)


# -------------------------
# Weather API Test
# -------------------------
def test_weather_api():
    """Test function to debug weather API"""
    print("🧪 Testing Weather API...")
    test_cities = ["Dhaka", "dhaka", "Dhaka,BD", "London"]
    for city in test_cities:
        try:
            result = get_weather(city)
            print(f"🌤️ {city}: {result}")
        except Exception as e:
            print(f"Error for {city}: {str(e)}")


# -------------------------
# Main Interactive Loop Refactor
# -------------------------
def handle_question(question: str):
    """Process a single question."""
    question_lower = question.lower()

    if question_lower in {'q', 'quit', 'exit'}:
        return "exit"

    if not question.strip():
        print("Please enter a question.")
        return None

    if question_lower == 'test weather':
        test_weather_api()
        return None

    print(f"\n🤔 Processing: {question}\n{'-' * 30}")
    try:
        response = agent_executor.invoke({"input": question})
        output = response.get('output')

        if output:
            print(f"\n✅ Final Answer:\n{output}")
            return None

        intermediate_steps = response.get('intermediate_steps', [])
        if intermediate_steps and len(intermediate_steps[-1]) > 1:
            print(f"\n⚠️ Agent result:\n{intermediate_steps[-1][1]}")
        else:
            print("\n❌ Agent stopped without providing an answer.")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


def main():
    """Run the interactive assistant."""
    print("🍕 Pizza Restaurant & Weather Assistant")
    print("Ask me about restaurants or weather information!")
    print("Type 'q' or 'quit' to exit.\n")

    while True:
        try:
            question = input("Ask your question: ").strip()
            result = handle_question(question)
            if result == "exit":
                print("Goodbye! 👋")
                break
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
