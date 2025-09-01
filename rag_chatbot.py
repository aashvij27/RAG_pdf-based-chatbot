import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PIL import Image
import fitz
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import asyncio
from langchain_community.vectorstores import Chroma


st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")


with st.sidebar:
    st.title("PDF Upload")
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.success("PDF uploaded successfully!")

        st.subheader("PDF Preview")
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        num_pages = len(pdf_document)
        page_num = st.number_input("Page", min_value=1, max_value=num_pages, value=1)
        
        page = pdf_document.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # type: ignore
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        st.image(img, caption=f"Page {page_num}", use_column_width=True)


        
st.title("RAG PDF Chatbot with Gemini")

if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text


if uploaded_file is not None:
    if st.session_state.chain is None:
        if "GOOGLE_API_KEY" not in os.environ:
            st.error("Please set your GOOGLE_API_KEY environment variable or use Streamlit secrets.")
        else:
            with st.spinner("Processing PDF..."):
                pdf_text = process_pdf(uploaded_file)
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                texts = text_splitter.split_text(pdf_text)
                
                metadatas = [{"source": f"chunk_{i}"} for i in range(len(texts))]
                
                
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.set_event_loop(asyncio.new_event_loop())
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
                
                message_history = ChatMessageHistory()
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    output_key="answer",
                    chat_memory=message_history,
                    return_messages=True,
                )
                
                st.session_state.chain = ConversationalRetrievalChain.from_llm(
                    ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.7),
                    chain_type="stuff",
                    retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
                    memory=memory,
                    return_source_documents=True,
                )
                
                st.success("PDF processed successfully!")


st.subheader("Chat with PDF!")
user_input = st.text_input("Ask a question about the uploaded pdf:")

if user_input:
    if st.session_state.chain is None:
        st.warning("Please upload a PDF file first.")
    else:
        with st.spinner("Formulating response..."):
            response = st.session_state.chain.invoke({"question": user_input})
            answer = response["answer"]
            source_documents = response["source_documents"]
            
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=answer))


chat_container = st.container()
with chat_container:
    for message in reversed(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            st.markdown(f'You: {message.content}')
        elif isinstance(message, AIMessage):
            st.markdown(f'Bot: {message.content}')
        
        if isinstance(message, AIMessage):
            with st.expander("View Sources"):
                for idx, doc in enumerate(source_documents):
                    st.write(f"Source {idx + 1}:", doc.page_content[:150] + "...")