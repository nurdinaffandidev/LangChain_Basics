# Import ChatOllama to run a local LLM using the Ollama backend
from langchain_ollama import ChatOllama

# Import ChatPromptTemplate to define a structured multi-message prompt
from langchain.prompts.chat import ChatPromptTemplate

# Initialize the chat model with Ollama using the "mistral" model
# Ensure it's downloaded with `ollama pull mistral` and the Ollama server is running
chat_model = ChatOllama(model="mistral")

# Define the system-level instruction (sets model behavior or personality)
# This tells the model to act as a translator from one language to another
system_template = "You are a helpful assistant that translates {input_language} to {output_language} commonly used."

# Define the human prompt — this is where the actual text to be translated goes
human_template = "{text}"

# Combine both system and human messages into a chat prompt template
# The `from_messages` method takes a sequence of role-message pairs
# - "system" is treated like setup or background context
# - "human" is the user’s actual message/input
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", human_template)
])

# Format the full list of messages by filling in the variables.
# The variables are passed as keyword arguments
# This generates a message list the model can understand in conversational form
messages = chat_prompt.format_messages(
    input_language="English",         # Sets {input_language} → English
    output_language="Bahasa Melayu",          # Sets {output_language} → Malay
    text="I am have not eaten today."   # Replaces {text} with the sentence to translate
)

# Invoke the model with the prepared prompt messages
# It processes the conversation and returns the response
result = chat_model.invoke(messages)

# Output the model's response text (translation)
print(result.content)
