from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

"""
llama index is base on OpenAI 
    -> you need to have OpenAI key
    -> in this project I use local ollama
data -> documents -> index -> query_engine
"""

# Configure Ollama
Settings.llm = Ollama(model="llama3.2:latest", request_timeout=120.0, temperature=0.1)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", embed_batch_size=10)
Settings.text_splitter = SentenceSplitter(chunk_size=1024)


documents = SimpleDirectoryReader(
    input_dir="./data"
).load_data()  # A list of Document objects that contain the text content and metadata from your files
# load documents and build index
index = VectorStoreIndex.from_documents(
    documents=documents
)  # create vector index from documents
# index = data structure for retrive relevant context from query
"""
^
|
1. documents chunking -> each documents is split into smaller text chunk -> retrieval more efficient and relevant. 
2. embedding generation -> each chunk is passes to the embedding model like Ollama, OpenAI, Gemini or HuggingFace
                        -> model convert text into high-dimentional vector(embedding)
*** Note: embedding = convert text into a vector -> then use cosine to calculate similarity

3. vector index construction -> all embedding are stored in a vector database(index)
4. metadata storage -> metadata (like documents ID, chunk position , of file name) is stored to help identity and retrieve the original text later.
5. ready for retrieval -> return VectorStoreIdex that contain all embeddings and metadata from the documents -> complex data structure
"""
query_engine = index.as_query_engine()
# use index as query engine, using query to ask question.
"""
^
|
1. Take your question as input.
2. Convert it into an embedding.
3. Search the vector index for the most relevant document chunks.
4. Pass those chunks to the LLM to generate a final answer.
"""
response = query_engine.query("what's the content of the introduction part?")
"""
1. Converts your question into an embedding (vector).
2. Searches the vector index for the most relevant document chunks using vector similarity.
3. Passes those chunks to the LLM (here, Ollama with "llama2") to generate a natural language answer.
4. Returns the answer as the response object.
"""
print(response)
