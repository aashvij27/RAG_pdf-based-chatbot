# PDF RAG Chatbot
A chatbot that answers queries from PDF documents using RAG (Retrieval-Augmented Generation).

## Tech Stack
- Streamlit
- LangChain + Ollama (LLaMA 3)
- Chroma Vector DB
- PyPDF2

## Features
- Upload PDF
- Extract + split text
- Ask questions, get contextual answers

## Run
```bash
pip install -r requirements.txt
streamlit run main.py
