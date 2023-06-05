# !pip install bs4
# !pip install nest_asyncio

# General


import os
import json
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from json import JSONDecodeError
from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.docstore.document import Document
import asyncio
import nest_asyncio

# Memory
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from embeddings.HuggingFaceEmbedding import hu

from langchain.tools.human.tool import HumanInputRun

# Tools
from contextlib import contextmanager
from typing import Optional
from langchain.agents import tool
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from tempfile import TemporaryDirectory
import HuggingFaceAPI

from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pydantic import Field
from langchain.chains.qa_with_sources.loading import (
    load_qa_with_sources_chain,
    BaseCombineDocumentsChain,
)

from embeddings.HuggingFaceEmbedding import newEmbeddingFunction
ROOT_DIR = TemporaryDirectory()


from langchain import ChatOpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
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




load_dotenv()
select_model = input(
    "Select the model you want to use (1, 2, 3 or 4) \n \
1) ChatGPT \n \
2) HuggingChat \n \
3) BingChat \n \
4) Google Bard \n \
5) HuggingFace \n \
>>> "
)

llm = ChatOpenAI(model_name="gpt-4", temperature=1.0)


@contextmanager
def pushd(new_dir):
    """Context manager for changing the current working directory."""
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)




# !pip install playwright
# !playwright install
async def async_load_playwright(url: str) -> str:
    """Load the specified URLs using Playwright and parse using BeautifulSoup."""
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright

    try:
        print(">>> WARNING <<<")
        print(
            "If you are running this for the first time, you nedd to install playwright"
        )
        print(">>> AUTO INSTALLING PLAYWRIGHT <<<")
        os.system("playwright install")
        print(">>> PLAYWRIGHT INSTALLED <<<")
    except:
        print(">>> PLAYWRIGHT ALREADY INSTALLED <<<")
        pass
    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            results = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results


def run_async(coro):
    event_loop = asyncio.get_event_loop()
    return event_loop.run_until_complete(coro)


@tool
def browse_web_page(url: str) -> str:
    """Verbose way to scrape a whole webpage. Likely to cause issues parsing."""
    return run_async(async_load_playwright(url))




def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=3000,
        chunk_overlap=20,
        length_function=len,
    )




class WebpageQATool(BaseTool):
    name = "query_webpage"
    description = (
        "Browse a webpage and retrieve the information relevant to the question."
    )
    text_splitter: RecursiveCharacterTextSplitter = Field(
        default_factory=_get_text_splitter
    )
    qa_chain: BaseCombineDocumentsChain

    def _run(self, url: str, question: str) -> str:
        """Useful for browsing websites and scraping the text information."""
        result = browse_web_page.run(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        # TODO: Handle this with a MapReduceChain
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i : i + 4]
            window_result = self.qa_chain(
                {"input_documents": input_docs, "question": question},
                return_only_outputs=True,
            )
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [
            Document(page_content="\n".join(results), metadata={"source": url})
        ]
        return self.qa_chain(
            {"input_documents": results_docs, "question": question},
            return_only_outputs=True,
        )

    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError


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


