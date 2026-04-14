from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

if os.environ['GROQ_API_KEY']:
    print("GROQ_API_KEY API Key is set.")
else:
    raise ValueError("GROQ_API_KEY API Key is not set.")


llm = ChatGroq(model="llama-3.3-70b-versatile")

# pydantic schema
from pydantic import BaseModel, Field

class graph_schema(BaseModel):
    topic: str = Field(description="The topic of the graph")
    post: str = Field(description="LinkedIn post content")
    curated_post: str = Field(description="Curated LinkedIn post content")

def create_post(state: graph_schema) -> graph_schema:
    topic = state.topic
    post = llm.invoke(f"Write a linkedin post on topic {topic}")
    state.post = post.content
    return state

def curate_post(state: graph_schema) -> graph_schema:
    post = state.post
    curated_post = llm.invoke(f"Curate the following post in genz tone {post}")
    state.curated_post = curated_post
    return state

from langgraph.graph import StateGraph, START, END

graph = StateGraph(graph_schema)

graph.add_node("create_post", create_post)
graph.add_node("curated_post", curate_post)


graph.add_edge(START, "create_post")
graph.add_edge("create_post", "curated_post")
graph.add_edge("curated_post", END)

pydantic_graph = graph.compile()

from IPython.display import Image, display

# print(pydantic_graph.get_graph().draw_mermaid())

response = pydantic_graph.invoke({
    "topic": "Comparison between the traditional RAG and graph RAG",
    "post": "",
    "curated_post": ""
})

print(response)


