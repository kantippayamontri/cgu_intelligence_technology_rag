# Build a RAG agent with LangChain - Docs by LangChain
Overview
--------

One of the most powerful applications enabled by LLMs is sophisticated question-answering (Q&A) chatbots. These are applications that can answer questions about specific source information. These applications use a technique known as Retrieval Augmented Generation, or [RAG](https://docs.langchain.com/oss/python/langchain/retrieval). This tutorial will show how to build a simple Q&A application over an unstructured text data source. We will demonstrate:

1.  A RAG [agent](#rag-agents) that executes searches with a simple tool. This is a good general-purpose implementation.
2.  A two-step RAG [chain](#rag-chains) that uses just a single LLM call per query. This is a fast and effective method for simple queries.

### Concepts

We will cover the following concepts:

*   **Indexing**: a pipeline for ingesting data from a source and indexing it. _This usually happens in a separate process._
*   **Retrieval and generation**: the actual RAG process, which takes the user query at run time and retrieves the relevant data from the index, then passes that to the model.

Once we’ve indexed our data, we will use an [agent](https://docs.langchain.com/oss/python/langchain/agents) as our orchestration framework to implement the retrieval and generation steps.

### Preview

In this guide we’ll build an app that answers questions about the website’s content. The specific website we will use is the [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post by Lilian Weng, which allows us to ask questions about the contents of the post. We can create a simple indexing pipeline and RAG chain to do this in ~40 lines of code. See below for the full code snippet:

Expand for full code snippet

```
import bs4
from langchain.agents import AgentState, create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain.messages import MessageLikeRepresentation
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Construct a tool for retrieving context
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)

```


```
query = "What is task decomposition?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

```


```
================================ Human Message =================================

What is task decomposition?
================================== Ai Message ==================================
Tool Calls:
  retrieve_context (call_xTkJr8njRY0geNz43ZvGkX0R)
 Call ID: call_xTkJr8njRY0geNz43ZvGkX0R
  Args:
    query: task decomposition
================================= Tool Message =================================
Name: retrieve_context

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Task decomposition can be done by...

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Component One: Planning...
================================== Ai Message ==================================

Task decomposition refers to...

```


Check out the [LangSmith trace](https://smith.langchain.com/public/a117a1f8-c96c-4c16-a285-00b85646118e/r).

Setup
-----

### Installation

This tutorial requires these langchain dependencies:

For more details, see our [Installation guide](https://docs.langchain.com/oss/python/langchain/install).

### LangSmith

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls. As these applications get more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent. The best way to do this is with [LangSmith](https://smith.langchain.com/). After you sign up at the link above, make sure to set your environment variables to start logging traces:

```
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."

```


Or, set them in Python:

```
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

```


### Components

We will need to select three components from LangChain’s suite of integrations. Select a chat model:

Select an embeddings model:

*   OpenAI
    
*   Azure
    
*   Google Gemini
    
*   Google Vertex
    
*   AWS
    
*   HuggingFace
    
*   Ollama
    
*   Cohere
    
*   MistralAI
    
*   Nomic
    
*   NVIDIA
    
*   Voyage AI
    
*   IBM watsonx
    
*   Fake
    

```
pip install -U "langchain-openai"

```


```
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

```


Select a vector store:

*   In-memory
    
*   AstraDB
    
*   Chroma
    
*   FAISS
    
*   Milvus
    
*   MongoDB
    
*   PGVector
    
*   PGVectorStore
    
*   Pinecone
    
*   Qdrant
    

```
pip install -U "langchain-core"

```


```
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

```


1\. Indexing
------------

Indexing commonly works as follows:

1.  **Load**: First we need to load our data. This is done with [Document Loaders](about:/oss/python/langchain/retrieval#document_loaders).
2.  **Split**: [Text splitters](about:/oss/python/langchain/retrieval#text_splitters) break large `Documents` into smaller chunks. This is useful both for indexing data and passing it into a model, as large chunks are harder to search over and won’t fit in a model’s finite context window.
3.  **Store**: We need somewhere to store and index our splits, so that they can be searched over later. This is often done using a [VectorStore](about:/oss/python/langchain/retrieval#vectorstores) and [Embeddings](about:/oss/python/langchain/retrieval#embedding_models) model.

![index_diagram](https://mintcdn.com/langchain-5e9cc07a/I6RpA28iE233vhYX/images/rag_indexing.png?fit=max&auto=format&n=I6RpA28iE233vhYX&q=85&s=21403ce0d0c772da84dcc5b75cff4451)

### Loading documents

We need to first load the blog post contents. We can use [DocumentLoaders](about:/oss/python/langchain/retrieval#document_loaders) for this, which are objects that load in data from a source and return a list of [Document](https://reference.langchain.com/python/langchain_core/documents/#langchain_core.documents.base.Document) objects. In this case we’ll use the [`WebBaseLoader`](https://docs.langchain.com/oss/python/integrations/document_loaders/web_base), which uses `urllib` to load HTML from web URLs and `BeautifulSoup` to parse it to text. We can customize the HTML -> text parsing by passing in parameters into the `BeautifulSoup` parser via `bs_kwargs` (see [BeautifulSoup docs](https://beautiful-soup-4.readthedocs.io/en/latest/#beautifulsoup)). In this case only HTML tags with class “post-content”, “post-title”, or “post-header” are relevant, so we’ll remove all others.

```
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

```


```
print(docs[0].page_content[:500])

```


```
      LLM Powered Autonomous Agents

Date: June 23, 2023  |  Estimated Reading Time: 31 min  |  Author: Lilian Weng


Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.
Agent System Overview#
In

```


**Go deeper** `DocumentLoader`: Object that loads data from a source as list of `Documents`.

*   [Integrations](https://docs.langchain.com/oss/python/integrations/document_loaders): 160+ integrations to choose from.
*   [`BaseLoader`](https://reference.langchain.com/python/langchain_core/document_loaders/#langchain_core.document_loaders.BaseLoader): API reference for the base interface.

### Splitting documents

Our loaded document is over 42k characters which is too long to fit into the context window of many models. Even for those models that could fit the full post in their context window, models can struggle to find information in very long inputs. To handle this we’ll split the [`Document`](https://reference.langchain.com/python/langchain_core/documents/#langchain_core.documents.base.Document) into chunks for embedding and vector storage. This should help us retrieve only the most relevant parts of the blog post at run time. As in the [semantic search tutorial](https://docs.langchain.com/oss/python/langchain/knowledge-base), we use a `RecursiveCharacterTextSplitter`, which will recursively split the document using common separators like new lines until each chunk is the appropriate size. This is the recommended text splitter for generic text use cases.

```
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

```


```
Split blog post into 66 sub-documents.

```


**Go deeper** `TextSplitter`: Object that splits a list of [`Document`](https://reference.langchain.com/python/langchain_core/documents/#langchain_core.documents.base.Document) objects into smaller chunks for storage and retrieval.

*   [Integrations](https://docs.langchain.com/oss/python/integrations/splitters)
*   [Interface](https://python.langchain.com/api_reference/text_splitters/base/langchain_text_splitters.base.TextSplitter.html): API reference for the base interface.

### Storing documents

Now we need to index our 66 text chunks so that we can search over them at runtime. Following the [semantic search tutorial](https://docs.langchain.com/oss/python/langchain/knowledge-base), our approach is to [embed](about:/oss/python/langchain/retrieval#embedding_models/) the contents of each document split and insert these embeddings into a [vector store](about:/oss/python/langchain/retrieval#vectorstores/). Given an input query, we can then use vector search to retrieve relevant documents. We can embed and store all of our document splits in a single command using the vector store and embeddings model selected at the [start of the tutorial](about:/oss/python/langchain/rag#components).

```
document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])

```


```
['07c18af6-ad58-479a-bfb1-d508033f9c64', '9000bf8e-1993-446f-8d4d-f4e507ba4b8f', 'ba3b5d14-bed9-4f5f-88be-44c88aedc2e6']

```


**Go deeper** `Embeddings`: Wrapper around a text embedding model, used for converting text to embeddings.

*   [Integrations](https://docs.langchain.com/oss/python/integrations/text_embedding): 30+ integrations to choose from.
*   [Interface](https://reference.langchain.com/python/langchain_core/embeddings/#langchain_core.embeddings.embeddings.Embeddings): API reference for the base interface.

`VectorStore`: Wrapper around a vector database, used for storing and querying embeddings.

*   [Integrations](https://docs.langchain.com/oss/python/integrations/vectorstores): 40+ integrations to choose from.
*   [Interface](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html): API reference for the base interface.

This completes the **Indexing** portion of the pipeline. At this point we have a query-able vector store containing the chunked contents of our blog post. Given a user question, we should ideally be able to return the snippets of the blog post that answer the question.

2\. Retrieval and Generation
----------------------------

RAG applications commonly work as follows:

1.  **Retrieve**: Given a user input, relevant splits are retrieved from storage using a [Retriever](about:/oss/python/langchain/retrieval#retrievers).
2.  **Generate**: A [model](https://docs.langchain.com/oss/python/langchain/models) produces an answer using a prompt that includes both the question with the retrieved data

![retrieval_diagram](https://mintcdn.com/langchain-5e9cc07a/I6RpA28iE233vhYX/images/rag_retrieval_generation.png?fit=max&auto=format&n=I6RpA28iE233vhYX&q=85&s=994c3585cece93c80873d369960afd44) Now let’s write the actual application logic. We want to create a simple application that takes a user question, searches for documents relevant to that question, passes the retrieved documents and initial question to a model, and returns an answer. We will demonstrate:

1.  A RAG [agent](#rag-agents) that executes searches with a simple tool. This is a good general-purpose implementation.
2.  A two-step RAG [chain](#rag-chains) that uses just a single LLM call per query. This is a fast and effective method for simple queries.

### RAG agents

One formulation of a RAG application is as a simple [agent](https://docs.langchain.com/oss/python/langchain/agents) with a tool that retrieves information. We can assemble a minimal RAG agent by implementing a [tool](https://docs.langchain.com/oss/python/langchain/tools) that wraps our vector store:

```
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

```


Given our tool, we can construct the agent:

```
from langchain.agents import create_agent


tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)

```


Let’s test this out. We construct a question that would typically require an iterative sequence of retrieval steps to answer:

```
query = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()

```


```
================================ Human Message =================================

What is the standard method for Task Decomposition?

Once you get the answer, look up common extensions of that method.
================================== Ai Message ==================================
Tool Calls:
  retrieve_context (call_d6AVxICMPQYwAKj9lgH4E337)
 Call ID: call_d6AVxICMPQYwAKj9lgH4E337
  Args:
    query: standard method for Task Decomposition
================================= Tool Message =================================
Name: retrieve_context

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Task decomposition can be done...

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Component One: Planning...
================================== Ai Message ==================================
Tool Calls:
  retrieve_context (call_0dbMOw7266jvETbXWn4JqWpR)
 Call ID: call_0dbMOw7266jvETbXWn4JqWpR
  Args:
    query: common extensions of the standard method for Task Decomposition
================================= Tool Message =================================
Name: retrieve_context

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Task decomposition can be done...

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Component One: Planning...
================================== Ai Message ==================================

The standard method for Task Decomposition often used is the Chain of Thought (CoT)...

```


Note that the agent:

1.  Generates a query to search for a standard method for task decomposition;
2.  Receiving the answer, generates a second query to search for common extensions of it;
3.  Having received all necessary context, answers the question.

We can see the full sequence of steps, along with latency and other metadata, in the [LangSmith trace](https://smith.langchain.com/public/7b42d478-33d2-4631-90a4-7cb731681e88/r).

### RAG chains

In the above [agentic RAG](#rag-agents) formulation we allow the LLM to use its discretion in generating a [tool call](about:/oss/python/langchain/models#tool-calling) to help answer user queries. This is a good general-purpose solution, but comes with some trade-offs:



* ✅ Benefits: Search only when needed – The LLM can handle greetings, follow-ups, and simple queries without triggering unnecessary searches.
  * ⚠️ Drawbacks: Two inference calls – When a search is performed, it requires one call to generate the query and another to produce the final response.
* ✅ Benefits: Contextual search queries – By treating search as a tool with a query input, the LLM crafts its own queries that incorporate conversational context.
  * ⚠️ Drawbacks: Reduced control – The LLM may skip searches when they are actually needed, or issue extra searches when unnecessary.
* ✅ Benefits: Multiple searches allowed – The LLM can execute several searches in support of a single user query.
  * ⚠️ Drawbacks: 


Another common approach is a two-step chain, in which we always run a search (potentially using the raw user query) and incorporate the result as context for a single LLM query. This results in a single inference call per query, buying reduced latency at the expense of flexibility. In this approach we no longer call the model in a loop, but instead make a single pass. We can implement this chain by removing tools from the agent and instead incorporating the retrieval step into a custom prompt:

```
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message


agent = create_agent(model, tools=[], middleware=[prompt_with_context])

```


Let’s try this out:

```
query = "What is task decomposition?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

```


```
================================ Human Message =================================

What is task decomposition?
================================== Ai Message ==================================

Task decomposition is...

```


In the [LangSmith trace](https://smith.langchain.com/public/0322904b-bc4c-4433-a568-54c6b31bbef4/r/9ef1c23e-380e-46bf-94b3-d8bb33df440c) we can see the retrieved context incorporated into the model prompt. This is a fast and effective method for simple queries in constrained settings, when we typically do want to run user queries through semantic search to pull additional context.

Returning source documents

The above [RAG chain](#rag-chains) incorporates retrieved context into a single system message for that run.As in the [agentic RAG](#rag-agents) formulation, we sometimes want to include raw source documents in the application state to have access to document metadata. We can do this for the two-step chain case by:

1.  Adding a key to the state to store the retrieved documents
2.  Adding a new node via a [pre-model hook](about:/oss/python/langchain/agents#pre-model-hook) to populate that key (as well as inject the context).

```
from typing import Any
from langchain_core.documents import Document
from langchain.agents.middleware import AgentMiddleware, AgentState


class State(AgentState):
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        retrieved_docs = vector_store.similarity_search(last_message.text)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        augmented_message_content = (
            f"{last_message.text}\n\n"
            "Use the following context to answer the query:\n"
            f"{docs_content}"
        )
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": retrieved_docs,
        }


agent = create_agent(
    model,
    tools=[],
    middleware=[RetrieveDocumentsMiddleware()],
)

```


Next steps
----------

Now that we’ve implemented a simple RAG application via [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent), we can easily incorporate new features and go deeper:

*   [Stream](https://docs.langchain.com/oss/python/langchain/streaming) tokens and other information for responsive user experiences
*   Add [conversational memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) to support multi-turn interactions
*   Add [long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory) to support memory across conversational threads
*   Add [structured responses](https://docs.langchain.com/oss/python/langchain/structured-output)
*   Deploy your application with [LangSmith Deployments](https://docs.langchain.com/langsmith/deployments)

* * *