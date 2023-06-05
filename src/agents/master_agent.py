from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.docstore.document import Document
import asyncio
import nest_asyncio
# Memory
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.tools.human.tool import HumanInputRun
# Tools
from contextlib import contextmanager
from typing import Optional
from langchain.agents import Tool
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from tempfile import TemporaryDirectory
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources.loading import (
    load_qa_with_sources_chain,
    BaseCombineDocumentsChain,
)
from embeddings.HuggingFaceEmbedding import newEmbeddingFunction
ROOT_DIR = TemporaryDirectory()
from master_tools.WebpageQATool import WebpageQATool
from langchain import ChatOpenAI

#ingest code -> embed the code -> set up qa langchain bot with the code
# embed stackoverflowPromptTemplate
# embed github
# add prompts as tools -> generate architecture, documentation
# add meta prompting
# Define your embedding model

embeddings_model = newEmbeddingFunction
embedding_size = 1536  # if you change this you need to change also in Embedding/HuggingFaceEmbedding.py
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})

llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)


query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))


# !pip install duckduckgo_search
web_search = DuckDuckGoSearchRun()

tools = [
    web_search,
    WriteFileTool(),
    ReadFileTool(),
    query_website_tool,
    HumanInputRun(), # Activate if you want the permit asking for help from the human
]


agent = AutoGPT.from_llm_and_tools(
    ai_name="Cognitive Coder",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever(search_kwargs={"k": 5}),
    human_in_the_loop=True, # Set to True if you want to add feedback at each step.
)


# agent.chain.verbose = True
agent.run([input("Enter the objective of the AI system: (Be realistic!) ")])