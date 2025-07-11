# Import the ChatOllama class from the langchain_ollama package.
# This allows LangChain to interact with LLMs served by Ollama locally.
from langchain_ollama import ChatOllama

# Initialize a chat model using a local Ollama model.
# Replace "mistral" with other models like "llama3", "phi3", "codellama", etc.
# The model must be downloaded already via `ollama pull mistral`
chat_model = ChatOllama(model="mistral")

# Send a prompt to the model using the `invoke()` method.
# This is a synchronous call that returns a single response.
response = chat_model.invoke("hi!")

# Print the full response object (usually a LangChain `AIMessage`).
print("\n Response:")
print("====================================")
print(response)

# Optionally, just print the text content of the response.
print("\n Response Content:")
print("====================================")
print(response.content)
