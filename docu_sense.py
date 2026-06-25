import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

load_dotenv()

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True}
    )

def load_and_chunk_pdfs(pdf_path: list) -> list:
    documents = []
    for path in pdf_path:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", "\t", ""],
        length_function=len
    )

    return splitter.split_documents(documents)

def build_vectorstore(chunks: list) -> FAISS:
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)

def build_rag_chain(vector_store: FAISS, api_key: str) -> ConversationalRetrievalChain:
    llm = ChatGroq(
        api_key = api_key,
        model_name = "llama-3.1-8b-instant",
        temperature = 0.2
    )

    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 4}
    )

    memory = ConversationBufferWindowMemory(
        k=10,
        memory_key = "chat_history",
        return_messages = True,
        output_key = "answer"
    )

    qa_prompt = PromptTemplate(
        input_variables = ["context", "question"],
        template = """You are a helpful document assistant. Answer only based on the provided context.
            If the answer is not in the context, say: "I don't have enough information in the uploaded documents to answer this."
            Never fabricate information.

            Context: {context}
            Question: {question}

            Answer:"""
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        memory = memory,
        return_source_documents = True,
        combine_docs_chain_kwargs = {"prompt": qa_prompt},
        output_key = "answer",
        verbose = False
    )

    return chain

def extract_sources(source_docs: list) -> list:
    seen = set()
    sources = []
    for doc in source_docs:
        file = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", 0) + 1
        key = (file, page)
        if key not in seen:
            seen.add(key)
            sources.append({"file": file, "page": page})
    return sources
