# 1. define tools and model
import json
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()

# SOLUTION: 1. Define model and embedding
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)

embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# SOLUTION: 2. Define state
class GraphState(TypedDict):
    """
    question: question from user
    generation: answer from LLM
    documents: Retrieve documents

    retrieve_attempt: number of retrieval attempts
    is_sufficient: whether the answer is sufficient
    rewritten_query: query rewritten for better retrieval
    """

    question: str
    generation: str
    documents: list[str]
    references: list[str]
    retrieve_attempts: int
    is_sufficient: bool
    rewritten_query: str


# SOLUTION: 3. Define Node
def node_retrieve(state: dict):
    """
    retrieve the documents base on the question
    """
    print("\n" + "=" * 60)
    print("📄 RETRIEVE DOCUMENTS")
    print("=" * 60)
    question = state["question"]
    print(f"  Question: {question}")

    # Optimized settings for tutorial/documentation markdown files
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Good for tutorial content
        chunk_overlap=150,  # 15% overlap to preserve context
        separators=[
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            " ",
            "",
        ],  # Split on markdown headers first
    )

    all_docs = []

    # Load Markdown files
    md_loader = DirectoryLoader(
        "./hw_data",
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,  # Show loading progress
        use_multithreading=True,  # Faster loading for multiple files
    )
    md_docs = md_loader.load()
    print(f"  Loaded {len(md_docs)} Markdown documents")

    # Show which files were loaded
    for doc in md_docs:
        filename = doc.metadata.get("source", "unknown")
        print(f"    - {filename}")

    all_docs.extend(md_docs)

    if not all_docs:
        print("  ✗ No documents found in ./hw_data folder")
        return {"documents": [], "references": []}

    # Split all documents into chunks
    docs = text_splitter.split_documents(all_docs)
    print(f"  Split into {len(docs)} chunks total")

    # using Chroma as the vector database
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embedding, persist_directory="./chroma_db"
    )
    # convert vector database into retriever
    retriever = vectordb.as_retriever(
        search_type="mmr",  # use MMR for diversity
        search_kwargs={
            "k": 3,
            "fetch_k": 10,  # Fetch more candidates, then select diverse ones
            "lambda_mult": 0.5,  # 0 = max diversity, 1 = max relevance
        },
    )  # use to try k=3 but got the same result 3 time -> the data is overlap

    # retrieve relavant documents
    documents = retriever.invoke(question)

    return {
        "documents": [doc.page_content for doc in documents],
        "references": ["local database" for i in range(len(documents))],
    }


# SOLUTION: check the data is sufficient or not
def node_is_sufficient(state: dict):
    print("\n" + "=" * 60)
    print("🔍 CHECKING INFORMATION SUFFICIENCY")
    print("=" * 60)

    # check relavancing using LLM
    relevance_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing relevance of retrieved documents to a user question. "
                "If the documents contain keywords or semantic meaning related to the question, grade it as relevant. "
                'Give a score of "yes", "no", or "ambiguous" to indicate whether the documents are relevant to the question. '
                '"ambiguous" means the documents may be partially relevant or you are unsure.',
            ),
            (
                "user",
                'Retrieved documents:\n{context}\n\nUser question: {question}\n\nAre these documents relevant? Answer only "yes", "no", or "ambiguous".',
            ),
        ]
    )

    question = state["question"]
    documents = state["documents"]

    # Check relevance using LLM
    # context = "\n\n".join([doc.page_content for doc in documents])
    context = "\n\n".join([doc for doc in documents])

    relavance_chain = relevance_prompt | llm
    response = relavance_chain.invoke({"context": context, "question": question})

    # Extract the answer
    is_relevant = response.content.strip().lower()
    print(f"  Relevance check: {is_relevant.upper()}")

    if "yes" in is_relevant:
        print("  ✓ Documents are relevant to the question")
        print(f"  Retrieved {len(documents)} document(s)")
        return {
            "documents": [doc for doc in documents],
            "retrieve_attempts": state.get("retrieve_attempts", 0) + 1,
            "is_sufficient": True,
        }
    elif "ambiguous" in is_relevant:
        print("  ⚠️ Documents may be partially relevant")
        print(f"  Retrieved {len(documents)} document(s)")
        print("  → Will supplement with internet search")
        return {
            "documents": [doc for doc in documents],  # Keep existing docs
            "retrieve_attempts": state.get("retrieve_attempts", 0) + 1,
            "is_sufficient": False,  # Search internet for more info
        }
    else:
        print("  ✗ Documents are NOT relevant to the question")
        return {
            "documents": [],
            "references": [],
            "retrieve_attempts": state.get("retrieve_attempts", 0) + 1,
            "is_sufficient": False,
        }


def node_check_sufficient(state: dict):
    """
    check the condition in is_sufficient
    """
    is_sufficient = state["is_sufficient"]
    if is_sufficient:
        print("  ✓ Local documents are sufficient")
        print("  → Proceeding to citation generation")
        return "check_documents"
    else:
        # in the case that find the data from the internet but not found any useful information
        if state["retrieve_attempts"] > 1:
            return "check_documents"

        # search further from the internet
        print("  ✗ Local documents insufficient")
        print("  → Searching the internet for additional information")
        return "search_internet"


# SOLUTION: search to the internet by using Tavily
def node_search_internet(state: dict):
    print("\n" + "=" * 60)
    print("🌐 SEARCHING INTERNET")
    print("=" * 60)
    tavily_tool = TavilySearch(k=5)  # k is the number of search results to return
    query = state["question"]
    print(f"  Query: {query}")
    results = tavily_tool.run(query)
    documents = state["documents"]
    references = state["references"]
    for r in results["results"]:
        documents.append(r["content"])
        references.append(r["url"])

    print(f"  ✓ Found {len(results['results'])} internet results")
    print(f"  Total documents: {len(documents)}")

    return {
        "documents": documents,
        "retrieve_attempts": state.get("retrieve_attempts", 0) + 1,
    }


# SOLUTION: find citation from documents to the answer
def node_find_citation(state: dict):
    print("\n" + "=" * 60)
    print("📝 GENERATING ANSWER WITH CITATIONS")
    print("=" * 60)
    question = state["question"]
    documents = state["documents"]
    print(f"  Question: {question}")
    print(f"  Using {len(documents)} document(s)")

    if not len(documents):
        print("  ✗ No documents available for citation")
        return END

    # find the citation
    # Create context with document numbers
    numbered_docs = [f"[{i+1}] {doc}" for i, doc in enumerate(documents)]
    context = "\n\n".join(numbered_docs)

    # Prompt for final answer with citations
    final_prompt = PromptTemplate(
        template="""You are an assistant providing well-cited answers.
Use the following numbered documents to answer the question.
Include citations in your answer using [1], [2], etc.

Question: {question}

Documents:
{context}

Provide a comprehensive answer with citations:""",
        input_variables=["question", "context"],
    )

    chain = final_prompt | llm
    response = chain.invoke({"question": question, "context": context})

    # Add references at the bottom
    references = state["references"]
    documents = state["documents"]
    answer_with_refs = (
        response.content + "\n\n" + "=" * 60 + "\n📚 REFERENCES:\n" + "=" * 60 + "\n"
    )
    for i, ref in enumerate(references, 1):
        answer_with_refs += f"[{i}] [Source from]: {ref}\n"
        answer_with_refs += (
            f"    Content: {documents[i-1][:100]}...\n\n"  # Show first 100 chars
        )

    return {"generation": answer_with_refs}


def check_documents(state: dict):
    """
    check is the documents for citation is exists or not
    """
    if len(state["documents"]):
        return state
    else:
        # No documents found from both local and internet search
        no_answer_msg = "\n" + "=" * 60 + "\n"
        no_answer_msg += "❌ NO ANSWER AVAILABLE\n"
        no_answer_msg += "=" * 60 + "\n"
        no_answer_msg += "Sorry, I couldn't find relevant information to answer your question.\n\n"
        
        if state.get("retrieve_attempts", 0) > 1:
            no_answer_msg += "🔍 Searched:\n"
            no_answer_msg += "  • Local database - No relevant documents found\n"
            no_answer_msg += "  • Internet search - No relevant results found\n"
        else:
            no_answer_msg += "🔍 Searched:\n"
            no_answer_msg += "  • Local database - No relevant documents found\n"
        
        no_answer_msg += "\n💡 Suggestions:\n"
        no_answer_msg += "  • Try rephrasing your question\n"
        no_answer_msg += "  • Use different keywords\n"
        no_answer_msg += "  • Ask a more specific question\n"
        
        return {"generation": no_answer_msg}


def nice_print(data: dict):
    formatted_data = json.dumps(data, ensure_ascii=False, indent=2)
    print(formatted_data)


def ask_question(question: str):
    # Initialize grpah
    agent_builder = StateGraph(GraphState)

    # Add node
    agent_builder.add_node("retrieve", action=node_retrieve)
    agent_builder.add_node("is_sufficient", action=node_is_sufficient)
    agent_builder.add_node("find_citation", action=node_find_citation)
    agent_builder.add_node("search_internet", action=node_search_internet)
    agent_builder.add_node("check_documents", action=check_documents)

    # Add edge
    agent_builder.add_edge(START, "retrieve")
    agent_builder.add_edge("retrieve", "is_sufficient")

    agent_builder.add_conditional_edges(
        "is_sufficient", node_check_sufficient, ["check_documents", "search_internet"]
    )
    agent_builder.add_edge("search_internet", "is_sufficient")
    agent_builder.add_edge("check_documents", "find_citation")

    # Compile the agent
    agent = agent_builder.compile()

    # Show the agent
    png_bytes = agent.get_graph(xray=True).draw_mermaid_png()

    # Save to file
    with open("hw_agent.png", "wb") as f:
        f.write(png_bytes)

    result = agent.invoke(
        {
            "question": question,
            "retrieve_attempts": 0,
            "documents": [],
            "generation": "",
            "is_sufficient": False,
            "rewritten_query": "",
        }
    )
    print("\n" + "=" * 60)
    print("✨ FINAL ANSWER")
    print("=" * 60)
    print(result["generation"])
    print("\n" + "=" * 60)

def app():
    print("\n" + "=" * 60)
    print("🤖 RAG QUESTION ANSWERING SYSTEM")
    print("=" * 60)
    print("Type 'exit', 'quit', or 'q' to stop\n")

    while True:
        question = input("💬 Enter your question: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("\n👋 Goodbye!")
            break

        if not question:
            print("⚠️  Please enter a valid question.\n")
            continue

        ask_question(question=question)
        print("\n")
    


if __name__ == "__main__":
    app()