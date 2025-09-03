import logging
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools import search_tool, wiki_tool, save_tool


# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Pydantic Schema ---
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: List[str]
    tools_used: List[str]


# --- LLM ---
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         You are a research assistant.

         - Use `search_web` when the query is about current events or recent news.  
         - Use `wiki_tool` when the query is about established knowledge or history.  
         - Use `save_text_to_file` only at the end, once you’ve compiled the final result.  

         Always output structured JSON in the required format:
         {format_instructions}
         """),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


# --- Tools ---
tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- Runner ---
def run_research(query: str, search_result: str = "", wiki_result: str = "") -> ResearchResponse:
    """Run research with Ollama and return structured ResearchResponse"""

    llm = ChatOllama(model="llama3", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
         You are a research assistant.

         - Use `search_web` when the query is about current events or recent news.  
         - Use `wiki_tool` when the query is about established knowledge or history.  
         - Use `save_text_to_file` only at the end, once you’ve compiled the final result.  

         Always output structured JSON in the required format:
         {format_instructions}
         """),
        ("human",
         """
         Here is the user query:
         {query}

         Here is the search result (if any):
         {search_result}

         Here is the Wikipedia result (if any):
         {wiki_result}

         Based on the above, provide a concise summary, list your sources, and mention which tools you used.
         """),
    ]).partial(
        query=query,
        search_result=search_result,
        wiki_result=wiki_result,
        format_instructions=parser.get_format_instructions()
    )

    chain = prompt | llm

    # Run the model
    raw_output = chain.invoke({}).content.strip()

    # Try to parse JSON
    try:
        data = json.loads(raw_output)

        # unwrap if Ollama wrapped inside "properties"
        if "properties" in data:
            data = data["properties"]

        return ResearchResponse(**data)

    except Exception as e:
        print("⚠️ Failed to parse LLM output, falling back to raw text.")
        return raw_output


if __name__ == "__main__":
    query = input("🔍 What can I help you with? ")
    try:
        response = run_research(query)
        print("\n--- Research Output ---")
        print(response.json(indent=2))
    except Exception as e:
        print(f"⚠️ Research failed: {e}")
