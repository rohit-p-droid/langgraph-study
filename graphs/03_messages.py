from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import pprint

load_dotenv()

if not os.environ["GROQ_API_KEY"]:
    raise ValueError("[-] API key not found")

llm = ChatGroq(model="llama-3.3-70b-versatile")

from langchain_core.messages import HumanMessage

# response = llm.invoke([HumanMessage(content="Hello, how are you?")])

# pprint.pprint(response)

from typing import TypedDict, Annotated, List
from operator import add

class graph_schema(TypedDict):
    messages_manual: List
    messages_auto: Annotated[List, add]

from langchain_core.messages import AIMessage

def create_post(state: graph_schema) -> graph_schema:
    message_manual = state["messages_manual"]
    response_manual = llm.invoke(message_manual).content
    response_manual_ai = AIMessage(content=response_manual)
    state["messages_manual"] = message_manual + [response_manual_ai]

    messagest_auto = state["messages_auto"]
    response_auto_ai = llm.invoke(messagest_auto).content
    response_auto_ai = AIMessage(content=response_auto_ai)
    state["messages_auto"] = [response_auto_ai]

    return state


def curated_post(state: graph_schema):
    message_manual = state["messages_manual"]
    response_manual = llm.invoke(message_manual).content
    response_manual_ai = AIMessage(content=response_manual)
    state["messages_manual"] = message_manual + [response_manual_ai]

    messagest_auto = state["messages_auto"]
    response_auto_ai = llm.invoke(messagest_auto).content
    response_auto_ai = AIMessage(content=response_auto_ai)
    state["messages_auto"] = [response_auto_ai]

    return state


from langgraph.graph import StateGraph, START, END

graph = StateGraph(graph_schema)

graph.add_node("create_post", create_post)
graph.add_node("curated_post", curated_post)

graph.add_edge(START, "create_post")
graph.add_edge("create_post", "curated_post")
graph.add_edge("curated_post", END)

messages_graph = graph.compile()
from IPython.display import Image, display


# Image(first_graph.get_graph().draw_mermaid_png())

# print(messages_graph.get_graph().draw_mermaid())

response = messages_graph.invoke({
    "messages_manual": [HumanMessage(content="Importance of AI agents.")],
    "messages_auto": [HumanMessage(content="Importance of AI agents.")]
})

pprint.pprint(response) 



