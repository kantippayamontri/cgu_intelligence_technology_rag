llamaindex with llama2
prompt can not access external data -> rag can
install llamaindex: uv add llama-index

install ollama so that we can use embedding and llm from local
1. install ollama: https://ollama.com/
2. install ollama integration: pip install llama-index-llms-ollama llama-index-embeddings-ollama
3. pull to llama model: ollama pull llama2
                        ollama pull llama3.2 #3B

langchain with ollama
1. install the library: 
    - pip install langchain langchain-ollama langchain_core langchain-community langchainhub langchain-chroma langchain-huggingface
    - pip install pypdf
    - pip install sentence_transformers # use in loading embedding for huggingface embedding model
2. 

#NOTE: 
SystemMessage and HumanMessage is difference
 - SystemMessage is use to tell the model how to act like "you are a math student", "you are a helpful assistant"
 - HumanMessage is a question from user