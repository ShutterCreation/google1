import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# Page Configuration setup
st.set_page_config(page_title="AI Document Analyst", page_icon="📄", layout="wide")

st.title("📄 AI Document Analyst & Q&A Bot")
st.write("Upload a PDF and ask questions! Powered by Google Gemini 1.5.")

# Sidebar for API Key and Uploads
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
    
    st.header("📂 Document Upload")
    pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type=["pdf"])
    process_button = st.button("Process Document")

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save Vector Store (FAISS)
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to handle user's question (DIRECT GEMINI CALL - No Chains)
def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Load the saved FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # Context-a direct-a extract panrom
    context = "\n".join([doc.page_content for doc in docs])
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    
    # Direct Prompt formulation
    prompt = f"""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in provided context just say, "Sorry, the answer is not available in the uploaded document." 
    Do not provide wrong answers.\n\n
    Context:\n {context}\n
    Question: \n{user_question}\n
    Answer:
    """
    
    response = model.invoke(prompt)
    
    st.write("**Answer:**")
    st.success(response.content)

# Logic Execution
if process_button:
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar.")
    elif not pdf_docs:
        st.warning("Please upload a PDF document first.")
    else:
        with st.spinner("Processing your document..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks, api_key)
            st.success("Document Processed Successfully! You can now ask questions.")

st.divider()

# Question Input UI
user_question = st.text_input("Ask a question based on the uploaded document:")

if user_question:
    if not api_key:
        st.error("Please enter your Gemini API Key first.")
    elif not os.path.exists("faiss_index"):
        st.error("Please upload and process a PDF document first.")
    else:
        user_input(user_question, api_key)