# from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

if os.environ["GROQ_API_KEY"]:
    print("[+] API key found")
else:
    raise ValueError("[-] API key not found")

llm = ChatGroq(model="llama-3.3-70b-versatile")

# response = llm.invoke("Hello, how are you?")

# print(response.content)

# define schema
from typing import TypedDict, List

class graph_schema(TypedDict):
    name: str
    msg: str

# create node functions
def welcome(schema: graph_schema) -> graph_schema:
    curr_name = schema["name"]
    curr_msg = schema["msg"]

    response = llm.invoke(f"Hello, I am {curr_name}! {curr_msg}")

    schema["msg"] = response.content

    return schema

# create state graph
from langgraph.graph import StateGraph, START, END

graph = StateGraph(graph_schema)
## adding nodes
graph.add_node("welcome", welcome)

## adding edges
graph.add_edge(START, "welcome")
graph.add_edge("welcome", END)

# compile graph
from IPython.display import Image, display

first_graph = graph.compile()

# Image(first_graph.get_graph().draw_mermaid_png())

# print(first_graph.get_graph().draw_mermaid())

response = first_graph.invoke({"name": "Ravi", "msg": "How are you?"})

print(response)