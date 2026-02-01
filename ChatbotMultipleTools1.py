## tools
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

#ARXIV
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Query arxiv papers")
print(arxiv.name)

arxiv.invoke("Attention is all you need")

#WIKIPEDIA
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki, description="Query wikipedia papers")
print(wiki.name)

wiki.invoke("What is the latest research on Quantam Computing")

##TAVILI TOOL
from langchain_community.tools.tavily_search import TavilySearchResults
tavily = TavilySearchResults()
tavily.invoke("Provide me the latest AI news")


#Integrating tools into WORKFLOW
from dotenv import load_dotenv

dotenv_path = r"D:\MTech Projects\LangGraph\LangGraph-venv\.env"
load_dotenv(dotenv_path)

import os


print(os.getcwd())

# Verify they are loaded correctly
print(os.getenv("TAVILY_API_KEY"))
print(os.getenv("GROQ_API_KEY"))


#os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
#os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


#Combining all tools into a list
tools = [arxiv, wiki, tavily]

##Initialize the LLM model
from langchain_groq import ChatGroq

llm = ChatGroq(model="qwen/qwen3-32b")
llm.invoke('What is AI ?')


#bind llm with tools
llm_with_tools = llm.bind_tools(tools = tools)

#Execute the call
llm_with_tools.invoke("WHat is the latest news on AI")

#Workflow
## State Schema
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage  ##Human Message or AI message
from typing import Annotated  ##labelling
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


### Entire Chatbot With LangGraph
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition


### Node definition
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

## Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)

graph = builder.compile()

#View
display(Image(graph.get_graph().draw_mermaid_png()))


#INFERENCING

messages = graph.invoke({"messages":"1706.03762"})
for m in messages['messages']:
    m.pretty_print()

messages = graph.invoke({"messages":"Hi, My name is Shweta"})
for m in messages['messages']:
    m.pretty_print()


query = """
Summarize the key ideas from the paper titled 'Attention Is All You Need'.Then, get me a brief overview of 'Transformer (machine learning model)'.
Finally, provide me with any related insights or data.
"""

messages = graph.invoke({"messages": [query]})

for m in messages['messages']:
    m.pretty_print()


