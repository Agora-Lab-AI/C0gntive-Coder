from langchain import ChatOpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory


class MetaAgent():
    def __init__(self):
        self.initalize_meta_agent()

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
        self.thinking_prompt = "You're Cognitive, an AI model devoted to helping Humans by any means necessary. You're purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time. The user has provided you with: {objective} complete this instruction BY ANY MEANS NECESSARY by considering the solutions you've had until now:\n\
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




def main(task, max_iters=3, max_meta_iters=5):
    failed_phrase = 'task failed'
    success_phrase = 'task succeeded'
    key_phrases = [success_phrase, failed_phrase]
    
    instructions = 'None'
    for i in range(max_meta_iters):
        print(f'[Episode {i+1}/{max_meta_iters}]')
        chain = MetaAgent.initialize_chain(instructions, memory=None)
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