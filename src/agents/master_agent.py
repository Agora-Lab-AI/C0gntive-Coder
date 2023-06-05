
 

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
from FreeLLM import ChatGPTAPI  # FREE CHATGPT API
from FreeLLM import HuggingChatAPI  # FREE HUGGINGCHAT API
from FreeLLM import BingChatAPI  # FREE BINGCHAT API
from FreeLLM import BardChatAPI  # FREE GOOGLE BARD API
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.docstore.document import Document
import asyncio
import nest_asyncio

# Memory
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from Embedding import HuggingFaceEmbedding  # EMBEDDING FUNCTION

from langchain.tools.human.tool import HumanInputRun

# Tools
import os
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
ROOT_DIR = TemporaryDirectory()


from langchain import ChatOpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
#ingest code -> embed the code -> set up qa langchain bot with the code
# embed stackoverflow
# embed github
# add prompts as tools -> generate architecture, documentation
# add meta prompting

# Define your embedding model
embeddings_model = HuggingFaceEmbedding.newEmbeddingFunction
embedding_size = 1536  # if you change this you need to change also in Embedding/HuggingFaceEmbedding.py
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})


# Needed synce jupyter runs an async eventloop
nest_asyncio.apply()
# [Optional] Set the environment variable Tokenizers_PARALLELISM to false to get rid of the warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

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



"""
Feat1: generateDockerImageName

Feat2: getCodeArchitectureExplanation

FEAT3: getArchitectureSummary

Feat4: Generate Code and submit Pull Requests

Feat5: Generate Documentation from code base

Feat6: Infinite Context Length => auto self scaling db parameters? An vector collection for every codebase?

Agent Architecture
Task Identification Agent: This agent will identify the task to be performed based on natural language processing.

Design and Architecture Agent: Based on the task, this agent will propose a suitable architecture or design.

Code Generation Agent: This agent will generate code based on the proposed design.

Testing Agent: This agent will write and run tests for the code.

Debugging Agent: This agent will debug the code if any issues are identified during the testing phase.

Optimization Agent: This agent will suggest and make changes to optimize the code.

Documentation Agent: This agent will automatically document the code and the system.


"""


def initialize_chain(instructions, memory=None):
    if memory is None:
        memory = ConversationBufferWindowMemory()
        memory.ai_prefix = "Cognitive"

    template = f"""
    Instructions: {instructions}
    {{{memory.memory_key}}}
    Human: {{human_input}}
    Cognitive:"""

    prompt = PromptTemplate(
        input_variables=["history", "human_input"], 
        template=template
    )

    chain = LLMChain(
        llm=ChatOpenAI(temperature=0), 
        prompt=prompt, 
        verbose=True, 
        memory=ConversationBufferWindowMemory(),
    )
    return chain



class MetaAgent():
    def __init__(self):
        self.initalize_meta_agent()

    def get_new_instructions(self, meta_output):
        new_instructions = meta_output.split("Instructions:")[-1]
        return new_instructions
    
    def update_prompt(self, chat_history, user_goal):
        chain = LLMChain(
            llm=self.LLM ,
            prompt=self.meta_prompt,
            verbose=True
        )
        meta_output = chain.run(chat_history=chat_history, old_instructions=self.thinking_prompt, objective=user_goal)
        #gte the new instructions from the meta output
        new_instructions = self.get_new_instructions(meta_output)
        print("New thinking instructions: ", new_instructions)
        variables_required = ["{old_thoughts}"]
        has_required_variables = all(var in variables_required for var in variables_required)
        if not has_required_variables:
            print("Instructions failed to mutate")
        else:
            self.thinking_prompt = new_instructions
    

    def initalize_meta_agent(self):
        self.thinking_prompt = "You're Athena, an AI model devoted to helping Humans by any means necessary. You're purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time. The user has provided you with: {objective} complete this instruction BY ANY MEANS NECESSARY by considering the solutions you've had until now:\n\
        \n'{old_thoughts}'\n\n Think about the next best step to achive {objective}.\
        If you already have enough thoughts to achieve the goal, start improving some of the steps and verify that you are perfectly meeting the goal '{objective}'.\n Next step:"

        meta_template="""
        You need to change the following solutions instructions;\n'{old_instructions}'\n To make the solutions directly solving the user objective '{objective}'

        Solutions instructions will be used by an AI assistant to direct it to create the thoughts to progress in achieving the user goal: '{objective}'.
        The Solutions instructions have to lead to thoughts that make the AI progress fast in totally achieving the user goal '{objective}'. The Solutions generated have to be sharp and concrete, and lead to concrete visible progress in achieving the user's goal.


        An AI model has just had the below interactions with a user, using the above solutions instructions to progress in achieve the user's goal. AI Model's generated thoughts don't lead to good enough progress in achieving: '{objective}'
        Your job is to critique the model's performance using the old solution instructions and then revise the instructions so that the AI 
        model would quickly and correctly respond in the future to concretely achieve the user goal.

        Old thinking instructions to modify:

        ###
        {old_instructions}
        ###
        The strings '{{old_thoughts}}' and the string '{{objective}}'  have to appear in the new instructions as they will respectively be used by the AI model to store it's old thoughts, and the user's goal when it runs that instruction

        AI model's interaction history with the user:

        ###
        {chat_history}
        ###

        Please reflect on these interactions.

        You should critique the models performance in this interaction in respect to why the solutions it gave aren't directly leading to achieving the user's goals. What could the AI model have done better to be more direct and think better?
        Indicate this with "Critique: ....

        You should then revise the Instructions so that Assistant would quickly and correctly respond in the future.
        The AI model's goal is to return the most reliable solution that leads to fast progressing in achieving the user's goal in as few interactions as possible.
        The solutions generated should not turn around and do nothing, so if you notice that the instructions are leading to no progress in solving the user goal, modify the instructions so it leads to concrete progress.
        The AI Assistant will only see the new Instructions the next time it thinks through the same problem, not the interaction
        history, so anything important to do must be summarized in the Instructions. Don't forget any important details in
        the current Instructions! Indicate the new instructions by "Instructions: ..."

        VERY IMPORTANT: The string '{{old_thoughts'}} and the string '{{objective}}' have to appear in the new instructions as they will respectively be used by the AI model to store it's old thoughts, and the user's goal when it runs that instruction
        """

        self.meta_prompt = PromptTemplate(
            input_variables=[ 'old_instructions', 'objective', 'chat_history'],
            template=meta_template
        )

        meta_chain = LLMChain(
            llm=ChatOpenAI(temperature=0), 
            prompt=self.meta_prompt, 
            verbose=True, 
        )

        return meta_chain
    
    def get_chat_history(chain_memory):
        memory_key = chain_memory.memory_key
        chat_history = chain_memory.load_memory_variables(memory_key)[memory_key]
        return chat_history

    def get_new_instructions(meta_output):
        delimiter = 'Instructions: '
        new_instructions = meta_output[meta_output.find(delimiter)+len(delimiter):]
        return new_instructions


if select_model == "1":
    CG_TOKEN = os.getenv("CHATGPT_TOKEN", "your-chatgpt-token")

    if CG_TOKEN != "your-chatgpt-token":
        os.environ["CHATGPT_TOKEN"] = CG_TOKEN
    else:
        raise ValueError(
            "ChatGPT Token EMPTY. Edit the .env file and put your ChatGPT token"
        )

    start_chat = os.getenv("USE_EXISTING_CHAT", False)
    if os.getenv("USE_GPT4") == "True":
        model = "gpt4"
    else:
        model = "default"

    if start_chat:
        chat_id = os.getenv("CHAT_ID")
        if chat_id == None:
            raise ValueError("You have to set up your chat-id in the .env file")
        llm = ChatGPTAPI.ChatGPT(
            token=os.environ["CHATGPT_TOKEN"], conversation=chat_id, model=model
        )
    else:
        llm = ChatGPTAPI.ChatGPT(token=os.environ["CHATGPT_TOKEN"], model=model)

elif select_model == "2":
    if not os.path.exists("cookiesHuggingChat.json"):
        raise ValueError(
            "File 'cookiesHuggingChat.json' not found! Create it and put your cookies in there in the JSON format."
        )
    cookie_path = Path() / "cookiesHuggingChat.json"
    with open("cookiesHuggingChat.json", "r") as file:
        try:
            file_json = json.loads(file.read())
        except JSONDecodeError:
            raise ValueError(
                "You did not put your cookies inside 'cookiesHuggingChat.json'! You can find the simple guide to get the cookie file here: https://github.com/IntelligenzaArtificiale/Free-Auto-GPT"
            )  
    llm = HuggingChatAPI.HuggingChat(cookiepath = str(cookie_path))

elif select_model == "3":
    if not os.path.exists("cookiesBing.json"):
        raise ValueError(
            "File 'cookiesBing.json' not found! Create it and put your cookies in there in the JSON format."
        )
    cookie_path = Path() / "cookiesBing.json"
    with open("cookiesBing.json", "r") as file:
        try:
            file_json = json.loads(file.read())
        except JSONDecodeError:
            raise ValueError(
                "You did not put your cookies inside 'cookiesBing.json'! You can find the simple guide to get the cookie file here: https://github.com/acheong08/EdgeGPT/tree/master#getting-authentication-required."
            )
    llm = BingChatAPI.BingChat(
        cookiepath=str(cookie_path), conversation_style="creative"
    )

elif select_model == "4":
    GB_TOKEN = os.getenv("BARDCHAT_TOKEN", "your-googlebard-token")

    if GB_TOKEN != "your-googlebard-token":
        os.environ["BARDCHAT_TOKEN"] = GB_TOKEN
    else:
        raise ValueError(
            "GoogleBard Token EMPTY. Edit the .env file and put your GoogleBard token"
        )
    cookie_path = os.environ["BARDCHAT_TOKEN"]
    llm = BardChatAPI.BardChat(cookie=cookie_path)

elif select_model == "5":
    llm = HuggingFaceAPI.HuggingFace()

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "your-huggingface-token")

if HF_TOKEN != "your-huggingface-token":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
else:
    raise ValueError(
        "HuggingFace Token EMPTY. Edit the .env file and put your HuggingFace token"
    )


@contextmanager
def pushd(new_dir):
    """Context manager for changing the current working directory."""
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)


@tool
def process_csv(
    csv_file_path: str, instructions: str, output_path: Optional[str] = None
) -> str:
    """Process a CSV by with pandas in a limited REPL.\
 Only use this after writing data to disk as a csv file.\
 Any figures must be saved to disk to be viewed by the human.\
 Instructions should be written in natural language, not code. Assume the dataframe is already loaded."""
    with pushd(ROOT_DIR):
        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:
            return f"Error: {e}"
        agent = create_pandas_dataframe_agent(llm, df, max_iterations=30, verbose=True)
        if output_path is not None:
            instructions += f" Save output to disk at {output_path}"
        try:
            result = agent.run(instructions)
            return result
        except Exception as e:
            return f"Error: {e}"


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
    process_csv,
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



def main(task, max_iters=3, max_meta_iters=5):
    failed_phrase = 'task failed'
    success_phrase = 'task succeeded'
    key_phrases = [success_phrase, failed_phrase]
    
    instructions = 'None'
    for i in range(max_meta_iters):
        print(f'[Episode {i+1}/{max_meta_iters}]')
        chain = initialize_chain(instructions, memory=None)
        output = chain.predict(human_input=task)
        for j in range(max_iters):
            print(f'(Step {j+1}/{max_iters})')
            print(f'Cognitive: {output}')
            print(f'Human: ')
            human_input = input()
            if any(phrase in human_input.lower() for phrase in key_phrases):
                break
            output = chain.predict(human_input=human_input)
        if success_phrase in human_input.lower():
            print(f'You succeeded! Thanks for playing!')
            return
        meta_chain = MetaAgent.initialize_meta_chain()
        meta_output = meta_chain.predict(chat_history=MetaAgent.get_chat_history(chain.memory))
        print(f'Feedback: {meta_output}')
        instructions = MetaAgent.get_new_instructions(meta_output)
        print(f'New Instructions: {instructions}')
        print('\n'+'#'*80+'\n')
    print(f'You failed! Thanks for playing!')