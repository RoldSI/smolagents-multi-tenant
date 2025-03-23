import os
from dotenv import load_dotenv
from smolagents import tool, OpenAIServerModel, CodeAgent

load_dotenv()

@tool
def addition(a: int, b: int) -> int:
    """
    Adds two integers. I.e., it computes a + b.
    Args:
        a: the first summand
        b: the second summand
    """
    return a+b

@tool
def modulo(a: int, b: int) -> int:
    """
    Computes a mod b.
    Args:
        a: the value to compute the modulo of
        b: the modulo value
    """
    return a%b

# model = OpenAIServerModel(
#     model_id="meta-llama/Llama-3.3-70B-Instruct",
#     api_base="https://fmapi.swissai.cscs.ch",
#     api_key=os.environ["LLM_API_KEY"]
# )
model = OpenAIServerModel(
    model_id="qwen2.5-0.5b-instruct-mlx",
    api_base="http://127.0.0.1:1234/v1"
)
# model = OpenAIServerModel(
#     model_id="Qwen/Qwen2.5-7B-Instruct-1M",
#     api_base="https://api.research.computer/v1"
# )

agent = CodeAgent(
    tools=[addition, modulo],
    # tools=[],
    model=model,
    max_steps=20,
    verbosity_level=2,
)

request = """
Compute ((23+234)%234)+3
"""

agent_output = agent.run(request)
print("Final output:")
print(agent_output)
