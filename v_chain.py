# Import the ChatOllama model class to connect with a local LLM through Ollama
from langchain_core.output_parsers.base import T
from langchain_ollama import ChatOllama
# Import the template builder for structured chat prompts
from langchain_core.prompts import ChatPromptTemplate
# Import base class for creating a custom output parser
from langchain_core.output_parsers import BaseOutputParser

# Initialize the chat model with Ollama using the "mistral" model
chat_model = ChatOllama(model="mistral")

# Create a custom parser that processes the model's raw output by splitting the response into a Python list
class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")

# Define the system-level instruction (sets model behavior or personality)
system_template = """
You are a helpful assistance who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more.
"""

# Define the human prompt — this is where the actual category is being input for the model to churn out list of items belonging to input category
human_template = "{category}"

# Combine both system and human messages into a chat prompt template
# The `from_messages` method takes a sequence of role-message pairs
# - "system" is treated like setup or background context
# - "human" is the user’s actual message/input
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", human_template)
])

# This is a full chain:
# 1. Takes a dictionary with the "category"
# 2. Formats the chat prompt
# 3. Sends it to the LLM via Ollama
# 4. Parses the output using CommaSeparatedListOutputParser
chain = chat_prompt | chat_model | CommaSeparatedListOutputParser()

# Invoke the chain by providing a category to generate a list of items
result = chain.invoke({"category":"colors"}) # e.g., input prompt: "colors"
# print(type(result))
print("Result")
print("===========================")
print(result)