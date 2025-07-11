# Import base class for creating a custom output parser
from langchain_core.output_parsers import BaseOutputParser
# Import the template builder for structured chat prompts
from langchain_core.prompts import ChatPromptTemplate
# Import the ChatOllama model class to connect with a local LLM through Ollama
from langchain_ollama import ChatOllama

# Initialize the chat model with Ollama using the "mistral" model
chat_model = ChatOllama(model="mistral")

# Create a custom parser that extracts just the final answer from the model's output
class AnswerOutputParser(BaseOutputParser):
    def parse(self, text: str):
        """Parse the output of an LLM call"""
        return text.strip().split("answer =")

# Define the system-level instruction (sets model behavior or personality)
system_template = """
You are a helpful assistant that solves math problems and shows your work.
Output each step then you must return the answer in the following format: "answer = <answer here>" only once. 
If there is more than one answer, please separate it by comma. Example as such: "answer = 1, 3"
If answer has decimals show it in 2 decimal places.
Make sure to output the answer in all lowercases and to have exactly one space and one equal sign following it.
"""

# Define the human prompt — this is where the actual math problem is being input for the model to be solved
human_template = "{problem}"

# Combine both system and human messages into a chat prompt template
# The `from_messages` method takes a sequence of role-message pairs
# - "system" is treated like setup or background context
# - "human" is the user’s actual message/input
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", human_template)
])

# Format the full list of messages by filling in the variables
# The variables are passed as keyword arguments
# This generates a message list the model can understand in conversational form
messages = chat_prompt.format_messages(problem="2x^2 - 5x + 3 = 0")
result = chat_model.invoke(messages)
print("Result content:")
print("===========================")
print(result.content, end="\n\n")

# Use your custom AnswerOutputParser to extract just the final answer
parsed = AnswerOutputParser().parse(result.content)
# print("Parsed result:")
# print("===========================")
# print(parsed, end="\n\n")
# print(type(parsed))

# converting parsed list to tuple to get answer
# steps, answer = parsed

# alternative: check length
answer = f"✅Answer: {parsed[-1]}" if len(parsed) == 2 else "Unable to get answer ‼️"

print("Answer:")
print("===========================")
print(answer)