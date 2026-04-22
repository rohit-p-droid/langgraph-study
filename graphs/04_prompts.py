from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import pprint

load_dotenv()

if not os.environ["GROQ_API_KEY"]:
    raise ValueError("[-] API key not found")

llm = ChatGroq(model="llama-3.3-70b-versatile")

from langchain_core.prompts import ChatPromptTemplate

my_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps to create and curate posts for social media"),
        ("human", "I want to create a post about {topic}")
    ]
)

chain = my_prompt | llm

response = chain.invoke({"topic": "Impotance of ai agents."})

print(response.content)