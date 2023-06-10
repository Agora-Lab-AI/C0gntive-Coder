from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT

# Memory
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.tools.human.tool import HumanInputRun
# Tools
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from tempfile import TemporaryDirectory
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains.qa_with_sources.loading import (
    load_qa_with_sources_chain,
)

from embeddings.HuggingFaceEmbedding import newEmbeddingFunction
ROOT_DIR = TemporaryDirectory()
from master_tools.WebpageQATool import WebpageQATool
from langchain import ChatOpenAI



embeddings_model = newEmbeddingFunction
embedding_size = 1536  
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