import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# 1. Correct File Path
pdf_path = "Data/Document.pdf"

# 2. Load and Chunk PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Embeddings (Free local model)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create Vector Store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Build RAG Chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Run Test Query
response = rag_chain.invoke({"input": "What is the main topic of this document?"})
print("Answer:", response["answer"])