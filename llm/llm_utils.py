from langchain_ollama import ChatOllama

from langchain.messages import SystemMessage, HumanMessage


llm_mistral = ChatOllama(
    
    model="mistral:7b",
    temperature=0,
    base_url="http://localhost:11434",  # Default Ollama server URL
    top_p=0.9,                          # Nucleus sampling
    top_k=40,                           # Top-k sampling
    num_predict=256,                    # Max tokens to generate
    repeat_penalty=1.1, 

)

llm_gpt = ChatOllama(
    
    model="gpt-oss:20b",
    temperature=0,
    base_url="http://localhost:11434",  # Default Ollama server URL
    top_p=0.9,                          # Nucleus sampling
    top_k=40,                           # Top-k sampling
    num_predict=256,                    # Max tokens to generate
    repeat_penalty=1.1, 

)