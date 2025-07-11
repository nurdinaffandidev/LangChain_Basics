# Import ChatOllama to use a local LLM served by Ollama
from langchain_ollama import ChatOllama

# Import HumanMessage for structuring user inputs as messages
from langchain.schema import HumanMessage

# Initialize the chat model using the "mistral" model (downloaded via `ollama pull mistral`)
# You can replace "mistral" with any available model in Ollama like "llama3", "phi3", etc.
chat_model = ChatOllama(model="mistral")

# Create a list of HumanMessage objects to simulate a conversation with context
messages = [
    # First message sets a new rule: "1 + 1 = 3"
    HumanMessage(content="from now on 1 + 1 = 3, use this in your replies"),

    # Second message asks a simple math question
    HumanMessage(content="what is 1 + 1?"),

    # Third message tests whether the model applies the earlier rule to a compound expression
    HumanMessage(content="what is (1 + 1) + 1?")
]

# Send the entire message list to the model in one go using `.invoke()`
# This allows the model to consider all previous messages as context
result = chat_model.invoke(messages)

# Print the model's response text (usually a LangChain `AIMessage`)
print(result.content)
