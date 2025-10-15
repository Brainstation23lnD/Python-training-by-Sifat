from ollama import Client

# =============================
# Simple Memory Node - 3
# =============================
class Memory:
    def __init__(self):
        self.history = []

    def add_user(self, text: str):
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.history.append({"role": "assistant", "content": text})

    def add_tool_observation(self, tool_name: str, observation: str):
        # FIX: removed extra bracket
        self.history.append({"role": "tool", "content": f"[{tool_name}] {observation}"})

    def chat_context(self, max_messages=10):
        # return last n messages
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.history[-max_messages:])


# =============================
# Tools Node 5
# =============================
class SerpAPITool:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def run(self, query: str):
        print(f"Running SerpAPITool with query: {query}")
        if not self.api_key:
            return "location: Mirpur, Dhaka; weather: Cloudy; temperature 81 F; humidity 80%; wind: 7mph; precipition:10%"
        return "real response here"


def calculator_tool(expression: str):
    print(f"Running calculator tool with query: {expression}")
    try:
        result = eval(expression, {"__builtins__": None})
        return str(result)
    except Exception as e:
        return f"Calculate Error: {str(e)}"


# =============================
# Call LLM wrapper 4 -- agents uses this
# =============================
def call_llm(prompt: str, use_mock=False):
    print("\n[LLM] Prompt (truncated):", prompt[:400].replace("\n", " | "), "\n")
    if use_mock:
        return "Answer: This is a mock answer"

    client = Client()
    response = client.chat(
        model="llama2",
        messages=[{"role": "user", "content": prompt}]
    )

    # FIX: handle both API shapes
    if "message" in response:
        return response["message"]["content"].strip()
    elif "messages" in response:
        return response["messages"][-1]["content"].strip()
    else:
        return str(response).strip()


# =============================
# Agent Node 4 with react loop
# =============================
class ReactAgent:
    def __init__(self, llm_func, tools: dict, memory: Memory, max_steps=4):
        self.memory = memory
        self.tools = tools
        self.llm_func = llm_func
        self.max_steps = max_steps

    def build_prompt(self, user_input: str, observation: str | None = None) -> str:
        mem = self.memory.chat_context()
        instructions = (
            "You are an agent that can call tools. When you need a tool, output:\n"
            "Tool: <tool_name>::<tool input>\n"
            "When you have final answer, output:\n"
            "Answer: <text>\n"
            "Tools available: " + ", ".join(self.tools.keys()) + "\n"
            "Make a short decision."
        )
        prompt = f"{mem}\nUser: {user_input}\n"
        if observation:
            prompt += f"\nObservation: {observation}\n"
        prompt += instructions
        return prompt

    @staticmethod
    def parse_llm_output(llm_out: str):
        llm_out = llm_out.strip()
        if llm_out.startswith("Answer:"):
            return "answer", llm_out.split("Answer:", 1)[1].strip()
        elif llm_out.startswith("Tool:"):
            try:
                _, rest = llm_out.split("Tool:", 1)
                tool_name, tool_input = rest.split("::", 1)
                return "tool", (tool_name.strip(), tool_input.strip())
            except Exception as e:
                return "error", f"Parse error of TOOL command: {str(e)}"
        else:
            return "answer", llm_out

    def run(self, user_query: str):
        self.memory.add_user(user_query)
        observation = None

        for _ in range(self.max_steps):
            prompt = self.build_prompt(user_query, observation)
            llm_out = self.llm_func(prompt)

            action, result = self.parse_llm_output(llm_out)

            if action == "answer":
                self.memory.add_assistant(result)
                return result

            elif action == "tool":
                tool_name, tool_input = result
                tool = self.tools.get(tool_name)
                if not tool:
                    observation = f"Tool {tool_name} not found."
                    self.memory.add_tool_observation("agent", observation)
                    continue

                print(f"Calling tool {tool_name} with input: {tool_input}")
                tool_result = tool.run(tool_input) if hasattr(tool, "run") else tool(tool_input)
                observation = str(tool_result)
                self.memory.add_tool_observation(tool_name, observation)

            elif action == "error":
                observation = result
                self.memory.add_tool_observation("agent", observation)

        # Fallback after max steps
        final_response = observation or "Sorry, I don't understand."
        self.memory.add_assistant(final_response)
        return final_response


# =============================
# Demo runner
# =============================
def demo_interaction(user_query: str, use_real_serpapi=False):
    memory = Memory()

    serp_tool = SerpAPITool(api_key="fake_api_key" if not use_real_serpapi else None)
    tools = {
        "SerpAPI": serp_tool,
        "Calculator": calculator_tool,
    }
    agent = ReactAgent(
        llm_func=call_llm,
        tools=tools,
        memory=memory
    )
    print(f"\n=== Agent interaction ===\nUser: {user_query}")
    user_query = user_query.strip()
    result = agent.run(user_query)
    # Postprocess
    print("\n--- Final Agent Response ---")
    print(result)
    print("\n--- Memory Trace ---")
    for item in memory.history:
        print(item["role"], ":", item["content"])


if __name__ == "__main__":
    # Examples:
    demo_interaction("What's the weather today in Mirpur Bangladesh?")
    print("\n\n--- Another example (calculator) ---\n")
    demo_interaction("What is 23 + 17?")
