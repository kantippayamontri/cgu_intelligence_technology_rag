from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document                 #for loading DOC
from langchain_community.document_loaders import PyPDFLoader         #for loading PDF
# from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# using Ollama
llm = OllamaLLM(model='llama3.2',temperature=0.1)
# set the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
# using PyPDFLoader to load the documents
loader = PyPDFLoader("./data/PreMed_Guide.pdf")
# load and split the document in the same time
docs = loader.load_and_split(text_splitter)
print(len(docs))

# using embedding model from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="maidalun1020/bce-embedding-base_v1")

# using Chroma as the vector database
vectordb = Chroma.from_documents(docs, embeddings)
# convert the vector database into retriever
retriever = vectordb.as_retriever()

# setup the prompt template
prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer the user\'s questions in English, based on the context provided below:\n\n{context}'),
    ('user', 'Question: {input}'),
])
# create the document chain, combile prompt and llm
document_chain = create_stuff_documents_chain(llm, prompt)
# create the retrieval chain，combine the retriever and the document chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

context = []
input_text = input('>>> ')
while input_text.lower() != 'bye':
    response = retrieval_chain.invoke({
        'input': input_text,
        'context': context
    })
    print(response['answer'])
    context = response['context']
    print("-------------------")
    print(response)
    input_text = input('>>> ')