#**AgenticAI**

## Overview
We have tried to create a multi tool agents workflow. First we have defined the tools like Arxiv, Wikipedia, Tavily. Then we have initialized ChatGroq model. Binding tools with the model is done. To define the workflow, we have defined the nodes first, built the graph, added nodes to the graph, added conditional edges to the graph and buit a multi agent Chatbot.

## Tech Stack
Python 
LangChain & LangGraph for agent orchestration
Groq 
Qwen (qwen3-32b) Large Language Model
ArXiv API (research paper retrieval)
Wikipedia API (knowledge lookup)
Tavily Search API (real-time web & AI news)
Tool-calling agents with LangGraph ToolNode
Jupyter / IPython for experimentation & graph visualization


## Problem Statement
Build a multi agent Chatbot leveraging LangGraph capabilities.

## Approach
Define the tools.
Initialize the Groq model
BInd the models with the tools
Create workflow by creating graph, defining nodes and conditional edges

## Results
Chatbot directing the chat to the right agent.


## How to Run
python ChatbotMultipleTools1.py
python prompty.py

