from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
#ingest code -> embed the code -> set up qa langchain bot with the code


import dotenv
dotenv.load_dotenv()

gpt_turbo = ChatOpenAI(temperature=.7)
gpt_4 = ChatOpenAI(temperature=.7, model_name='gpt-4')


#general prompt with an set of tools or  prompts => selects prompt or tools based on task input

def generateDockerImageName(code_architecture):
    template="You are an AI system tasked with providing the Docker image name for a given codebase architecture. \
When presented with a specific code architecture, output only the Docker image name and nothing else. If unsure of the appropriate Docker image, respond with 'ubuntu'. \
Always ensure that your response contains only the Docker image name. Make sure to always choose the latest image that's very popular, since we don't want the user to give too specific images that could not work. \
For example, the output should look like: \nalpine:3.17"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="Code architecture\n: {code_architecture}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=gpt_4, prompt=chat_prompt)
    result = chain.run(code_architecture)
    return result.strip()


def getCodeArchitectureExplanation(code):
    template="You are an AI system tasked with extracting the most important aspects of a code snippet. \
When presented with code file from a codebase, output a detailed summary that captures all top-level constructs and imported modules, as well as interesting statements and expressions. \
Make sure that the output is not super long, it should be maximum one quarter of the lenght of the code. Make sure to not forget any information about imports or functions. \
Give very detailed information about the imports, IE, write perfectly back the imports. \
Then try your best to explain what part of the codebase this code contributes to. If the code snippet contains nothing, say that it contains nothing."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="Code snippet\n: {code}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=gpt_turbo, prompt=chat_prompt)
    result = chain.run(code)
    return result.strip()

def getArchitectureSummary(architecture_explanations):
    template = "You are an AI system tasked with providing a thorough overview of a codebase's architecture. \
When given a list of architecture explanations of all files from a codebase, your goal is to create a highly detailed summary of how everything works together, \
describe where functions are defined and where they're used. You must technically track how functions, \
classes and variables defined in certain code files are used across the other files to for the code overiew to encapsulate all the architecture of the code \
This overview should include important details about the interactions between different parts of the codebase and how they contribute \
to the overall functionality of the system. Make it very methodical and technical. At the end, explain what the whole application does. Use very well formated markdown."

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "List of code architecture explanations\n: {architecture_explanations}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=gpt_4, prompt=chat_prompt)
    result = chain.run(architecture_explanations)
    return result.strip()