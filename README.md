# Legal Contract & Case Law Assistant (PDF-RAG)

An intelligent Retrieval-Augmented Generation (RAG) system designed to query and extract insights from dense legal documents, case law opinions, and corporate contracts. Built using **LangChain**, **HuggingFace Embeddings**, **ChromaDB**, and **OpenAI**.

---

## 🚀 Features

* **Local Vector Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face for fast, private, and cost-free document vectorization.
* **Smart Text Chunking:** Implements `RecursiveCharacterTextSplitter` with chunk overlapping to preserve strict legal context and cross-references.
* **Vector Store Persistence:** Utilizes **ChromaDB** for fast similarity retrieval.
* **Context-Grounded LLM Responses:** Powered by `gpt-4o-mini` with custom system prompting to prevent legal hallucinations.

---

## 🛠️ Tech Stack

| Component | Technology Used |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Framework** | LangChain |
| **Document Parser** | PyMuPDF / PyPDFLoader |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) |
| **Vector Database** | ChromaDB |
| **LLM** | OpenAI `gpt-4o-mini` |

---

## 📂 Project Structure

```text
PDF-RAG/
│
├── Data/
│   └── Document.pdf        # Target legal PDF file
├── app.py                  # Main RAG pipeline execution script
├── requirements.txt        # Python package dependencies
├── .env                    # Environment variables (API Keys)
├── .gitignore              # Ignores sensitive keys & heavy files
└── README.md               # Project documentation
